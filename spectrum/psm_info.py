import logging
import os
import numpy as np
import random
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pyteomics import mass
from typing import Tuple

from spectrum.labeling import (
    HeavyType,
    MASS_DELTA_C13_C12,
    MASS_DELTA_N15_N14,
    get_fixed_heavy_atom_counts,
    get_heavy_increase_mass,
    get_silac_increase_mass,
    has_label_site,
    parse_heavy_type,
    supports_modified_peptide,
)


PROTON_MASS = 1.00727646677


@dataclass(frozen=True)
class IsotopologueTarget:
    """One exact-mass contributor to a nominal isotope channel."""

    nominal_shift: int
    mass_shift: float
    relative_abundance: float


# 定义在全局，就不用频繁初始化了
_UNIMOD_XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unimod.xml",
)
with open(_UNIMOD_XML_PATH, 'rb') as f:
    unimods = mass.Unimod(source=f)


class PSMInfo:
    """ 记录一个 psm 的主要信息"""

    def __init__(
        self,
        sequence: str,
        charge: int,
        modify: [(int, int)],
        rt: np.float32,
        precursor_mz: np.float32,
        raw_title: str,
        protein_names: str,
        q_value: float | None = None,
        score: float | None = None,
        label_type: str | None = None,
        query_id: str | None = None,
        parent_id: str | None = None,
        group_id: str | None = None,
        pair_id: str | None = None,
        candidate_family_id: str | None = None,
        peptide_group_id: str | None = None,
    ):

        self._sequence = sequence
        self._charge = charge
        self._modify = modify
        self._rt = rt
        self._precursor_mz = precursor_mz
        self._raw_title = raw_title
        self._protein_names = protein_names
        self._q_value = q_value
        self._score = score
        self._label_type = label_type
        self._query_id = query_id
        self._parent_id = parent_id
        self._group_id = group_id
        self._pair_id = pair_id
        self._candidate_family_id = candidate_family_id
        self._peptide_group_id = peptide_group_id

    def to_dict(self):
        """将对象转为 JSON 兼容的字典"""
        d = {
            "sequence": self._sequence,
            "charge": self._charge,
            "modify": [list(pair) for pair in self._modify],  # tuple -> list
            # np.float32 -> float
            "rt": float(self._rt),
            "precursor_mz": float(self._precursor_mz),
            "raw_title": self._raw_title,
            "protein_names": self._protein_names,
        }
        if self._q_value is not None:
            d["q_value"] = float(self._q_value)
        if self._score is not None:
            d["score"] = float(self._score)
        if self._label_type is not None:
            d["label_type"] = self._label_type
        for name in (
                "query_id", "parent_id", "group_id", "pair_id",
                "candidate_family_id", "peptide_group_id"):
            value = getattr(self, f"_{name}")
            if value is not None:
                d[name] = value
        return d

    @classmethod
    def from_dict(cls, data: dict):
        """从字典重建 PSMInfo 对象，对新字段做 None 兜底以兼容老 JSON"""
        return cls(
            sequence=data["sequence"],
            charge=data["charge"],
            modify=[(int(pos), int(mod))
                    for pos, mod in data["modify"]],  # list -> tuple
            rt=np.float32(data["rt"]),
            precursor_mz=np.float32(data["precursor_mz"]),
            raw_title=data["raw_title"],
            protein_names=data["protein_names"],
            q_value=data.get("q_value"),
            score=data.get("score"),
            label_type=data.get("label_type"),
            query_id=data.get("query_id"),
            parent_id=data.get("parent_id"),
            group_id=data.get("group_id"),
            pair_id=data.get("pair_id"),
            candidate_family_id=data.get("candidate_family_id"),
            peptide_group_id=data.get("peptide_group_id"),
        )

    def __repr__(self):
        """ 实现标准输出 """
        return (f"PSMInfo(sequence='{self._sequence}', charge={self._charge}, "
                f"modify='{self._modify}', rt={self._rt}, "
                f"precursor_mz={self._precursor_mz}, raw_title='{
                    self._raw_title}')"
                f"protein_names={self._protein_names}")

    def get_key(self) -> Tuple[str, int, Tuple[Tuple[int, int], ...]]:
        """ 返回这个key ，两两 psm 中，sequence、charge、modify 相同认为是相同的。"""
        # 将 modify list 转为嵌套 tuple，使其可哈希
        mod_tuple = tuple(tuple(pair) for pair in self._modify)
        return (self._sequence, self._charge, mod_tuple)

    def get_key_with_raw(self) -> Tuple[str, int, Tuple[Tuple[int, int], ...], str]:
        """ 返回包含 raw_title 的 key。

        与 get_key() 不同，此 key 区分同一肽段在不同 raw 文件中的观测。
        用于 extract_common 等需要保留跨 raw 独立观测的场景。
        """
        mod_tuple = tuple(tuple(pair) for pair in self._modify)
        return (self._sequence, self._charge, mod_tuple, self._raw_title)

    def valid(self) -> bool:
        """
        检查自己是否合法，或者说受当前代码支持
        1. 氨基酸出现未知 X
        2. 待发现，之后再在这里进行添加
        """
        if 'X' in self._sequence:
            return False

        return True

    def get_SILAC_precursor_mz(self):
        """
        根据轻序列计算出重标重量
        主要是将 K：C(-6)13C(6) 和 R : C(-6)N(-4)13C(6)15N(4)
        """
        # 这是一个占位符，替换为你的实际计算方法
        # 可以根据修饰信息、氨基酸序列等来调整母离子质量
        heavy_mass = self._precursor_mz * self._charge

        heavy_mass += get_SILAC_increase_mass(self._sequence)

        return heavy_mass / self._charge

    def _assert_heavy_supported(self, heavy_type: HeavyType) -> None:
        """C13/N15 全代谢标记下，修饰基团里的 C/N 原子是否重标取决于
        原子来源与引入时机，当前 PSM 修饰表示并不携带这项信息。
        替换，但当前未实现。带修饰肽段走该路径会得到静默错误的重标质量，
        因此显式抛错。SILAC 只标记 K/R，不涉及修饰，故不受影响。"""
        if heavy_type in (HeavyType.C13, HeavyType.N15) and self._modify:
            raise NotImplementedError(
                "C13/N15 重标尚未支持带修饰的肽段："
                "修饰基团中的 C/N 原子标记状态未知。"
                f"sequence={self._sequence!r}, modify={self._modify!r}")

    def get_uniform_label_precursor_mz(self, heavy_type: HeavyType):
        """
        根据轻序列计算出重标重量，根据C和N两种不同的
        """

        # 修饰基团的原子来源/引入时机未知；上游策略应先过滤，底层仍保留
        # 显式守卫，避免绕过工作流时返回错误质量。
        self._assert_heavy_supported(heavy_type)
        heavy_mass = self._precursor_mz * self._charge

        heavy_mass += get_heavy_increase_mass(self._sequence, heavy_type)

        return heavy_mass / self._charge

    def get_C_N_HEAVY_precursor_mz(self, heavy_type: HeavyType):
        """Deprecated alias for :meth:`get_uniform_label_precursor_mz`."""
        return self.get_uniform_label_precursor_mz(heavy_type)

    def get_modify_mass(self, end_idx):
        """ 返回 从 [0,endix] 这个区间的修饰质量 """
        all_mass = 0

        for idx, mid in self._modify:
            if idx <= end_idx:
                tot_modify_mass = unimods.by_id(mid)["mono_mass"]

                all_mass += tot_modify_mass

        return all_mass

    def get_fragment_ions(self, heavy_type: HeavyType):
        """返回两个列表：b_ions, y_ions"""

        # 同 get_uniform_label_precursor_mz —— 修饰原子在 C13/N15
        # 下未被重标，带修饰肽段会得到静默错误的碎片质量，先显式失败。
        self._assert_heavy_supported(heavy_type)
        b_ions = []
        y_ions = []

        n = len(self._sequence)

        all_modify_mass = self.get_modify_mass(n)
        # 遍历所有长度，得到所有b,y离子
        for i in range(1, n):
            # 先获得b离子
            b_mass = mass.fast_mass(self._sequence[0:i], ion_type='b')
            b_mass += self.get_modify_mass(i-1)

            b_heavy_mass = (b_mass +
                            get_heavy_increase_mass(
                                self._sequence[0:i], heavy_type))

            b_ions.append((b_mass, b_heavy_mass))

            # 获得 y 离子
            y_mass = mass.fast_mass(self._sequence[-i:], ion_type='y')
            y_mass += all_modify_mass - self.get_modify_mass(n-i-1)

            y_heavy_mass = (y_mass +
                            get_heavy_increase_mass(
                                self._sequence[-i:], heavy_type))
            y_ions.append((y_mass, y_heavy_mass))

        b_ans = [("b", i+1, light_mass, heavy_mass)
                 for i, (light_mass, heavy_mass) in enumerate(b_ions)]
        y_ans = [("y", i+1, light_mass, heavy_mass)
                 for i, (light_mass, heavy_mass) in enumerate(y_ions)]
        return b_ans, y_ans

    def get_heavy_info(self, heavy_type: HeavyType):
        # TODO: 待实现，先逐步支持吧，一种一种重标考虑，考虑时尽量考虑通用性

        if heavy_type == HeavyType.SILAC:
            heavy_precursor_mz = self.get_SILAC_precursor_mz()
        else:
            heavy_precursor_mz = self.get_uniform_label_precursor_mz(heavy_type)

        b_ions, y_ions = self.get_fragment_ions(heavy_type)

        return heavy_precursor_mz, b_ions + y_ions


def get_SILAC_increase_mass(sequence: str):
    """ 计算这个序列中SILAC 重标应该增加多少重量 """
    return get_silac_increase_mass(sequence)


def _residual_isotope_composition(
    sequence: str,
    modifications=(),
    heavy_type: HeavyType = HeavyType.SILAC,
) -> mass.Composition:
    """Return atoms that retain their terrestrial isotope distribution."""
    selected = parse_heavy_type(heavy_type)
    if modifications and not supports_modified_peptide(selected):
        raise NotImplementedError(
            "ideal full-label C13/N15 isotope envelopes currently support "
            "unmodified peptides only")

    comp = mass.Composition(sequence)
    for _, modification_id in modifications:
        try:
            modification = unimods.by_id(int(modification_id))
            mod_comp = modification["composition"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"unknown or composition-less Unimod id {modification_id}"
            ) from exc
        comp += mod_comp

    for element, fixed_count in get_fixed_heavy_atom_counts(
            sequence, selected).items():
        comp[element] -= fixed_count

    if any(float(count) < 0 for count in comp.values()):
        raise ValueError(f"invalid isotope composition for {sequence!r}: {comp}")
    return comp


def get_residual_isotopologue_targets(
    sequence: str,
    modifications=(),
    heavy_type: HeavyType = HeavyType.SILAC,
) -> dict[int, list[IsotopologueTarget]]:
    """Return exact-mass targets for ideal-full-label M0/M1/M2 channels.

    Fixed label atoms are excluded at 100% purity/incorporation.  Residual
    natural-isotope contributors retain their distinct exact masses: e.g.
    15N is not searched at the 13C offset.  M+2 includes direct +2 isotopes
    (18O/34S) and every pair of +1 substitutions.  Relative abundances are
    expressed against the all-monoisotopic composition.
    """
    comp = _residual_isotope_composition(
        sequence, modifications, heavy_type)
    one_step = []
    direct_two = []
    for element in ("C", "H", "N", "O", "S"):
        count = int(comp.get(element, 0))
        if count <= 0:
            continue
        isotopes = mass.nist_mass[element]
        natural = [
            (number, values[0], values[1])
            for number, values in isotopes.items()
            if number != 0 and values[1] > 0
        ]
        base_number, base_mass, base_abundance = max(
            natural, key=lambda item: item[2])
        for number, isotope_mass, abundance in natural:
            nominal = int(number - base_number)
            if nominal not in (1, 2):
                continue
            item = {
                "element": element,
                "count": count,
                "nominal": nominal,
                "mass_shift": float(isotope_mass - base_mass),
                "ratio": float(abundance / base_abundance),
            }
            (one_step if nominal == 1 else direct_two).append(item)

    targets = {
        0: [IsotopologueTarget(0, 0.0, 1.0)],
        1: [],
        2: [],
    }
    for item in one_step:
        targets[1].append(IsotopologueTarget(
            1, item["mass_shift"], item["count"] * item["ratio"]))
    for item in direct_two:
        targets[2].append(IsotopologueTarget(
            2, item["mass_shift"], item["count"] * item["ratio"]))
    for left_index, right_index in combinations_with_replacement(
            range(len(one_step)), 2):
        left = one_step[left_index]
        right = one_step[right_index]
        if left_index == right_index:
            multiplicity = left["count"] * (left["count"] - 1) / 2
        else:
            multiplicity = left["count"] * right["count"]
        if multiplicity <= 0:
            continue
        targets[2].append(IsotopologueTarget(
            2,
            left["mass_shift"] + right["mass_shift"],
            multiplicity * left["ratio"] * right["ratio"],
        ))

    # Different elemental combinations can have the same exact mass to the
    # precision relevant here. Merge them for a compact extraction panel.
    for nominal in (1, 2):
        merged: dict[float, float] = {}
        for target in targets[nominal]:
            key = round(target.mass_shift, 9)
            merged[key] = merged.get(key, 0.0) + target.relative_abundance
        targets[nominal] = [
            IsotopologueTarget(nominal, shift, abundance)
            for shift, abundance in sorted(merged.items())
            if abundance > 0
        ]
    return targets


def get_theoretical_isotope_ratios(
    sequence: str,
    modifications=(),
    heavy_type: HeavyType = HeavyType.SILAC,
) -> list[float]:
    """Return ideal-full-label nominal [M0, M1, M2] ratios."""
    targets = get_residual_isotopologue_targets(
        sequence, modifications, heavy_type)
    envelope = np.asarray([
        sum(target.relative_abundance for target in targets[nominal])
        for nominal in (0, 1, 2)
    ], dtype="f8")

    if envelope[0] <= 0 or not np.all(np.isfinite(envelope)):
        raise ValueError(f"invalid theoretical isotope envelope: {envelope}")
    return (envelope / envelope[0]).tolist()


def get_isotopologue_mz_targets(
    heavy_m0_mz: float,
    charge: int,
    sequence: str,
    modifications=(),
    heavy_type: HeavyType = HeavyType.SILAC,
) -> dict[int, list[float]]:
    """Convert the residual exact-mass model into precursor m/z panels."""
    if int(charge) <= 0:
        raise ValueError("precursor charge must be positive")
    targets = get_residual_isotopologue_targets(
        sequence, modifications, heavy_type)
    return {
        nominal: [
            float(heavy_m0_mz) + target.mass_shift / int(charge)
            for target in values
        ]
        for nominal, values in targets.items()
    }


def sequence_controlled_shuffle(peptide, anchor_len=2, shuffle_ratio=0.5,
                                 seed=None, max_tries=10):
    """
    anchor_len=1: 保留C端K/R（标准做法）
    anchor_len=2: 保留C端"XK"或"XR"（保留y1+y2离子）

    seed: int or None. If provided, use a fresh random.Random(seed)
        instance for deterministic shuffle. If None, use module-level
        random (backward compat, non-deterministic).
        (P2-4, Pipeline-I2, 2026-06-03 audit.)

    保证打乱后序列与原序列不同（否则 feature_type=2 负样本=正样本，标签污染）：
    n_shuffle 至少为 2，并重试至多 max_tries 次直到 core 改变；当 core 无法产生
    不同排列（长度<2 或只有一种字符）时原样返回，调用方应跳过该负样本。
    """
    rng = random.Random(seed) if seed is not None else random

    # 安全检查：anchor_len 不能超过肽段长度
    anchor_len = min(anchor_len, len(peptide) - 1)  # 至少留1个字符用于shuffle

    core = peptide[:-anchor_len]   # 可shuffle部分
    anchor = peptide[-anchor_len:]  # C端锚定部分（通常是"K"或"R"）

    # core 无法打乱出不同序列：原样返回（调用方应跳过）
    if len(core) < 2 or len(set(core)) < 2:
        return peptide

    # 至少打乱 2 个位置才可能改变；重试直到 core 确实变化
    n_shuffle = min(max(2, int(len(core) * shuffle_ratio)), len(core))
    for _ in range(max_tries):
        indices = rng.sample(range(len(core)), n_shuffle)
        sel = [core[i] for i in indices]
        if len(set(sel)) < 2:
            continue   # 选中位置全是同一字符，换一组
        # 打乱选中值直到排列与原来不同（选中子集含 ≥2 种字符，必然可达）
        shuffled_vals = sel[:]
        for _ in range(20):
            rng.shuffle(shuffled_vals)
            if shuffled_vals != sel:
                break
        chars = list(core)
        for idx, val in zip(indices, shuffled_vals):
            chars[idx] = val
        candidate = ''.join(chars)
        if candidate != core:
            return candidate + anchor

    return peptide   # 多次仍未变（极少见）：原样返回，调用方应跳过
