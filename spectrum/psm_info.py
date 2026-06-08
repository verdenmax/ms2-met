import logging
import os
import numpy as np
import random
from enum import Enum
from pyteomics import mass
from typing import Tuple


# UniMod-canonical heavy-isotope mass deltas
MASS_DELTA_C13_C12 = 1.003355   # ¹³C − ¹²C
MASS_DELTA_N15_N14 = 0.997035   # ¹⁵N − ¹⁴N
PROTON_MASS = 1.00727646677


# 定义在全局，就不用频繁初始化了
_UNIMOD_XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unimod.xml",
)
with open(_UNIMOD_XML_PATH, 'rb') as f:
    unimods = mass.Unimod(source=f)


class HeavyType(Enum):
    SILAC = 1
    CHEAVY = 2
    NHEAVY = 3


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
        """CHEAVY/NHEAVY 全代谢标记下，修饰基团里的 C/N 原子也应被 13C/15N
        替换，但当前未实现。带修饰肽段走该路径会得到静默错误的重标质量，
        因此显式抛错。SILAC 只标记 K/R，不涉及修饰，故不受影响。"""
        if heavy_type in (HeavyType.CHEAVY, HeavyType.NHEAVY) and self._modify:
            raise NotImplementedError(
                "CHEAVY/NHEAVY 重标尚未支持带修饰的肽段："
                "修饰基团中的 C/N 原子未做 13C/15N 重标 (TODO)。"
                f"sequence={self._sequence!r}, modify={self._modify!r}")

    def get_C_N_HEAVY_precursor_mz(self, heavy_type: HeavyType):
        """
        根据轻序列计算出重标重量，根据C和N两种不同的
        """

        # TODO: 实现修饰原子的重标。全 13C/15N 代谢标记(CHEAVY/NHEAVY)下，
        # 修饰基团里的 C/N 原子同样应被 13C/15N 替换，但 get_heavy_increase_mass
        # 只统计了序列骨架/侧链原子，未覆盖修饰原子。带修饰肽段在此路径下会得到
        # 静默错误的重标质量，故先抛 NotImplementedError 让其显式失败。
        self._assert_heavy_supported(heavy_type)
        heavy_mass = self._precursor_mz * self._charge

        heavy_mass += get_heavy_increase_mass(self._sequence, heavy_type)

        return heavy_mass / self._charge

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

        # TODO: 同 get_C_N_HEAVY_precursor_mz —— 修饰原子在 CHEAVY/NHEAVY
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
            heavy_precursor_mz = self.get_C_N_HEAVY_precursor_mz(heavy_type)

        b_ions, y_ions = self.get_fragment_ions(heavy_type)

        return heavy_precursor_mz, b_ions + y_ions


def get_SILAC_increase_mass(sequence: str):
    """ 计算这个序列中SILAC 重标应该增加多少重量 """
    increase_mass = 0

    # 遍历肽段序列中的每个氨基酸
    for amino_acid in sequence:
        if amino_acid == 'K':  # 赖氨酸，加上 8.014204 C(-6)13C(6)N(-2)15N(2)
            increase_mass += 8.014204
        elif amino_acid == 'R':  # 精氨酸，加上 10.008275  C(-6)N(-4)13C(6)15N(4)
            increase_mass += 10.008275
    return increase_mass


def get_heavy_increase_mass(
    sequence: str,
    heavy_type: HeavyType,
) -> float:
    """ 根据重标类型，计算出这个 sequence 增加了多少质量"""

    if heavy_type == HeavyType.SILAC:
        return get_SILAC_increase_mass(sequence)

    increase_mass = 0
    composition = mass.Composition(sequence)

    if heavy_type == HeavyType.CHEAVY:
        increase_mass += composition['C'] * MASS_DELTA_C13_C12
    elif heavy_type == HeavyType.NHEAVY:
        increase_mass += composition['N'] * MASS_DELTA_N15_N14

    return increase_mass


def get_theoretical_isotope_ratios(sequence: str) -> list:
    """计算肽段的理论同位素分布比例 [M0, M1, M2]（Poisson 近似）。
    基于各元素的重同位素天然丰度，用 Poisson 模型估算 M+1/M+2 相对于 M0 的比值。
    """
    comp = mass.Composition(sequence)
    # 各元素对 M+1 峰的贡献（主要重同位素天然丰度）
    lam = (comp.get('C', 0) * 0.01109 +   # 13C
           comp.get('H', 0) * 0.000115 +  # 2H
           comp.get('N', 0) * 0.00364 +   # 15N
           comp.get('O', 0) * 0.00038 +   # 17O
           comp.get('S', 0) * 0.0079)     # 33S
    # Poisson 近似: P(k) ∝ λ^k / k!
    return [1.0, lam, lam * lam / 2.0]


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
