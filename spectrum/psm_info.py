
import numpy as np


class PSMInfo:
    """ 记录一个 psm 的主要信息"""

    def __init__(
        self,
        sequence: str,
        charge: int,
        modify: str,
        rt: np.float32,
        precursor_mz: np.float32,
        raw_title: str,
    ):

        self._sequence = sequence
        self._charge = charge
        self._modify = modify
        self._rt = rt
        self._precursor_mz = precursor_mz
        self._raw_title = raw_title
