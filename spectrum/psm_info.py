
import logging
import numpy as np
from enum import Enum
from pyteomics import mass


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
        rt_start: np.float32,
        rt_stop: np.float32,
        precursor_mz: np.float32,
        raw_title: str,
    ):

        self._sequence = sequence
        self._charge = charge
        self._modify = modify
        self._rt = rt
        self._rt_start = rt_start
        self._rt_stop = rt_stop
        self._precursor_mz = precursor_mz
        self._raw_title = raw_title

    def __repr__(self):
        """ 实现标准输出 """
        return (f"PSMInfo(sequence='{self._sequence}', charge={self._charge}, "
                f"modify='{self._modify}', rt={self._rt}, "
                f"rt_start= {self._rt_start}, rt_end={self._rt_stop},"
                f"precursor_mz={self._precursor_mz}, raw_title='{self._raw_title}')")

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

    def get_C_N_HEAVY_precursor_mz(self, heavy_type: HeavyType):
        """
        根据轻序列计算出重标重量，根据C和N两种不同的
        """

        # NOTE: 注意这里还要实现修饰的重标
        # TODO: 实现修饰的重标，这里重点是是否需要重标
        heavy_mass = self._precursor_mz * self._charge

        heavy_mass += get_heavy_increase_mass(self._sequence, heavy_type)

        return heavy_mass / self._charge

    def get_fragment_ions(self, heavy_type: HeavyType):
        """返回两个列表：b_ions, y_ions"""

        b_ions = []
        y_ions = []

        n = len(self._sequence)
        # 遍历所有长度，得到所有b,y离子
        for i in range(1, n):
            # NOTE: 这里还没有添加修饰
            # TODO: 修饰同上
            # 先获得b离子
            b_mass = mass.fast_mass(self._sequence[0:i], ion_type='b')

            b_heavy_mass = (b_mass +
                            get_heavy_increase_mass(
                                self._sequence[0:i], heavy_type))

            b_ions.append((b_mass, b_heavy_mass))

            # 获得 y 离子
            y_mass = mass.fast_mass(self._sequence[-i:], ion_type='y')

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
        increase_mass += composition['C'] * 1.00727646677
    elif heavy_type == HeavyType.NHEAVY:
        increase_mass += composition['N'] * 0.997036

    return increase_mass
