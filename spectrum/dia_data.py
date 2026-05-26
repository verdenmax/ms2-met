
import re
import logging
import numpy as np
import pandas as pd

from pyteomics import mzml
from spectrum.spectrum_utils import match_peak_ppm


DEFAULT_VALUE_NO_MOBILITY = 1e-6


def deduplicate_with_tolerance(arr, tolerance=0.1):
    """
    对float32数组进行容差去重并排序

    Args:
        arr: np.ndarray[float32] 输入数组
        tolerance: float 容差值

    Returns:
        去重并排序后的数组
    """
    if arr is None or len(arr) == 0:
        return None

    # 确保是float32类型
    arr = arr.astype(np.float32)

    # 先排序
    sorted_arr = np.sort(arr)

    # 容差去重
    unique_values = []
    for value in sorted_arr:
        if not unique_values or abs(value - unique_values[-1]) >= tolerance:
            unique_values.append(value)

    return np.array(unique_values, dtype=np.float32)


def _load_attrs(obj, data):
    """辅助函数：从 npz 数据填充属性"""
    # 标量
    obj.has_mobility = bool(data['has_mobility'])
    obj.has_ms1 = bool(data['has_ms1'])
    obj._max_mz_value = float(
        data['_max_mz_value']) if '_max_mz_value' in data else None
    obj._min_mz_value = float(
        data['_min_mz_value']) if '_min_mz_value' in data else None
    obj._zeroth_frame = int(data['_zeroth_frame'])
    obj._scan_max_index = int(data['_scan_max_index'])
    obj.frame_max_index = int(
        data['frame_max_index']) if 'frame_max_index' in data and data['frame_max_index'] is not None else None

    # 数组（自动是 mmap 视图）
    obj.ms1_indexs = data['ms1_indexs']
    obj.ms1_indexs_rt = data['ms1_indexs_rt']
    obj.ms2_indexs = data['ms2_indexs']
    obj.ms2_indexs_rt = data['ms2_indexs_rt']
    obj.precursor_scan_ids = data['precursor_scan_ids']
    obj._mz_values = data['_mz_values']
    obj.rt_values = data['rt_values']
    obj._intensity_values = data['_intensity_values']
    obj.mobility_values = data['mobility_values']
    obj._cycle_left_precursor = data['_cycle_left_precursor']

    # 可选数组
    obj._quad_max_mz_value = data['_quad_max_mz_value'] if '_quad_max_mz_value' in data else None
    obj._quad_min_mz_value = data['_quad_min_mz_value'] if '_quad_min_mz_value' in data else None
    obj._scan_id_to_index = data['_scan_id_to_index'] if '_scan_id_to_index' in data else None
    obj._peak_start_idx_list = data['_peak_start_idx_list'] if '_peak_start_idx_list' in data else None
    obj._peak_stop_idx_list = data['_peak_stop_idx_list'] if '_peak_stop_idx_list' in data else None
    obj._precursor_lower_mz = data['_precursor_lower_mz'] if '_precursor_lower_mz' in data else None
    obj._precursor_upper_mz = data['_precursor_upper_mz'] if '_precursor_upper_mz' in data else None


class DIAData:
    def __init__(self):
        # 初始化这个 dia 数据的所有特征
        """
        数据特征标识
        """
        self.has_mobility: bool = False
        self.has_ms1: bool = True

        """
        记录的原始数据关键数组, mz_value、rt_value、intensity_value、mobility_values。
        """
        # 记录所有的 ms1 index
        self.ms1_indexs: np.ndarray[tuple[int], np.dtype[np.int32]] = None
        self.ms1_indexs_rt: np.ndarray[tuple[int], np.dtype[np.float32]] = None
        # 记录所有的 ms2 index
        self.ms2_indexs: np.ndarray[tuple[int], np.dtype[np.int32]] = None
        self.ms2_indexs_rt: np.ndarray[tuple[int], np.dtype[np.float32]] = None
        # 这个ms2 index 对应的 ms1 信息
        self.precursor_scan_ids: np.ndarray[tuple[int],
                                            np.dtype[np.int32]] = None
        self._mz_values: np.ndarray[tuple[int],
                                    np.dtype[np.float32]] | None = None
        self.rt_values: np.ndarray[tuple[int],
                                   np.dtype[np.float32]] | None = None
        self._intensity_values: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)

        # TODO: 未使用
        self.mobility_values: (
            np.ndarray[tuple[int], np.dtype[np.float32]]) = np.array(
            [DEFAULT_VALUE_NO_MOBILITY, 0], dtype=np.float32
        )

        """ DIA 窗口相关属性，为了判断ms2 是否落在了同一个窗口 """
        self._cycle_left_precursor: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)

        """ mz 范围信息 """
        self._max_mz_value: np.float32 | None = None
        self._min_mz_value: np.float32 | None = None

        # TODO: 未使用
        self._quad_max_mz_value: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)
        self._quad_min_mz_value: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)

        """ 索引和边界信息 """
        self._scan_id_to_index: (
            np.ndarray[tuple[int], np.dtype[np.int32]] | None) = (None)
        self._peak_start_idx_list: (
            np.ndarray[tuple[int], np.dtype[np.int32]] | None) = (None)
        self._peak_stop_idx_list: (
            np.ndarray[tuple[int], np.dtype[np.int32]] | None) = (None)
        self._precursor_lower_mz: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)
        self._precursor_upper_mz: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)

        self._zeroth_frame: int = 0
        self._scan_max_index: int = 1
        self.frame_max_index: int | None = None

    # 在 DIAData 类中添加
    def save_to_file(self, filepath: str):
        """将所有 NumPy 数组和标量保存到 .npz 文件"""

        data = {
            # 标量属性
            'has_mobility': self.has_mobility,
            'has_ms1': self.has_ms1,
            '_max_mz_value': self._max_mz_value,
            '_min_mz_value': self._min_mz_value,
            '_zeroth_frame': self._zeroth_frame,
            '_scan_max_index': self._scan_max_index,
            'frame_max_index': self.frame_max_index,

            # 数组属性（只保存非 None 的）
            'ms1_indexs': self.ms1_indexs,
            'ms1_indexs_rt': self.ms1_indexs_rt,
            'ms2_indexs': self.ms2_indexs,
            'ms2_indexs_rt': self.ms2_indexs_rt,
            'precursor_scan_ids': self.precursor_scan_ids,
            '_mz_values': self._mz_values,
            'rt_values': self.rt_values,
            '_intensity_values': self._intensity_values,
            'mobility_values': self.mobility_values,
            '_cycle_left_precursor': self._cycle_left_precursor,
            '_quad_max_mz_value': self._quad_max_mz_value,
            '_quad_min_mz_value': self._quad_min_mz_value,
            '_scan_id_to_index': self._scan_id_to_index,
            '_peak_start_idx_list': self._peak_start_idx_list,
            '_peak_stop_idx_list': self._peak_stop_idx_list,
            '_precursor_lower_mz': self._precursor_lower_mz,
            '_precursor_upper_mz': self._precursor_upper_mz,
        }

        # 过滤掉 None 值（np.savez 不支持 None）
        data = {k: v for k, v in data.items() if v is not None}
        np.savez_compressed(filepath, **data)
        logging.info(f"Saved DIAData to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str, use_mmap: bool = True):
        """从 .npz 文件加载 DIAData，支持内存映射（只读）"""
        obj = cls()

        if use_mmap:
            # 使用 mmap_mode='r' 实现零拷贝共享
            with np.load(filepath, mmap_mode='r') as data:
                _load_attrs(obj, data)
        else:
            # 普通加载（用于主进程预处理）
            data = np.load(filepath)
            _load_attrs(obj, data)

        return obj

    def _get_retention_time(self, spectrum) -> float:
        """从谱图中提取保留时间（转换为秒）"""

        if 'scanList' in spectrum:
            scan = spectrum['scanList']['scan'][0]
            if 'scan start time' in scan:
                rt = scan['scan start time']
                return float(rt)
        return 0.0

    def _extract_scan_number(self, scan_id_str):
        """
        从谱图 ID 字符串中提取 scan number（整数）。
        例如："controllerType=0 controllerNumber=1 scan=1234" -> 1234
        """
        if scan_id_str is None:
            return -1

        match = re.search(r'scan=(\d+)', scan_id_str)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"无法从 scan_id 提取扫描号: {scan_id_str}")

    def _preallocate_arrays(self, total_spectra: int, total_peaks: int):
        """ 预先分配数组信息 """
        # 谱图信息数组
        self.precursor_scan_ids = np.zeros(total_spectra, dtype=np.int64)
        self.rt_values = np.zeros(total_spectra, dtype=np.float32)
        self._peak_start_idx_list = np.zeros(total_spectra, dtype=np.int64)
        self._peak_stop_idx_list = np.zeros(total_spectra, dtype=np.int64)
        self._precursor_lower_mz = np.zeros(total_spectra, dtype=np.float32)
        self._precursor_upper_mz = np.zeros(total_spectra, dtype=np.float32)

        # 峰数据数组
        self._mz_values = np.zeros(total_peaks, dtype=np.float32)
        self._intensity_values = np.zeros(total_peaks, dtype=np.float32)

        # 其他数组
        self._scan_id_to_index = np.zeros(total_spectra + 10, dtype=np.int64)

    def _process_single_spectrum(
        self, spectrum,
        spectrum_idx, current_peak_index
    ):
        """ 处理单个的谱图，将其中信息记录起来 """

        # 获取保留时间 (转换为秒)
        rt = self._get_retention_time(spectrum)

        # 获取质谱的scan id，不需要使用 spectrum_idx
        scan_id = self._extract_scan_number(spectrum['id'])

        # 获取spec title
        spec_title = spectrum.get('spectrum title', None).split()[0]

        # 获取 MS 级别
        ms_level = spectrum.get('ms level', 1)

        # 获取前体信息 (对于 MS2)
        precursor_scan_id = -1
        precursor_mz = None
        precursor_charge = None
        precursor_intensity = None
        isolation_lower = None
        isolation_upper = None

        if ms_level > 1 and 'precursorList' in spectrum:
            precursors = spectrum['precursorList']['precursor']
            if precursors:
                precursor = precursors[0]
                precursor_scan_id = self._extract_scan_number(
                    precursor.get('spectrumRef', None))
                selected_ions = precursor['selectedIonList']['selectedIon']

                if selected_ions:
                    precursor_mz = selected_ions[0].get(
                        'selected ion m/z', None)
                    precursor_charge = selected_ions[0].get(
                        'charge state', None)
                    precursor_intensity = selected_ions[0].get(
                        'peak intensity', None)

                # 获取隔离窗口
                if 'isolationWindow' in precursor:
                    isolation_lower = precursor['isolationWindow'].get(
                        'isolation window lower offset', 0)
                    isolation_upper = precursor['isolationWindow'].get(
                        'isolation window upper offset', 0)

            if precursor_mz is not None:
                isolation_lower = precursor_mz - isolation_lower
                isolation_upper = precursor_mz + isolation_upper

        # 检查是否有离子迁移率数据
        if 'scanList' in spectrum:
            scan = spectrum['scanList']['scan'][0]
            if 'ion mobility drift time' in scan:
                self.has_mobility = True

        # 获取 m/z 和强度数组
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array']

        # 记录谱图信息
        # _spectrum_info = {
        #     'spec_idx': spectrum_idx,
        #     'scan_id': scan_id,
        #     'rt': rt,
        #     'spec_title': spec_title,
        #     'ms_level': ms_level,
        #     'precursor_scan_id': precursor_scan_id,
        #     'precursor_mz': precursor_mz,
        #     'precursor_charge': precursor_charge,
        #     'precursor_intensity': precursor_intensity,
        #     'isolation_lower_mz': isolation_lower,
        #     'isolation_upper_mz': isolation_upper,
        #     'peak_start_idx': current_peak_index,
        #     'peak_stop_idx': current_peak_index + len(mz_array)
        # }

        del spectrum

        # 检查是否有 MS1 数据
        if ms_level == 1:
            self.has_ms1 = True

        """
        记录的原始数据关键数组, mz_value、rt_value、intensity_value、mobility_values。
        """
        peak_stop_idx = current_peak_index + len(mz_array)
        self.precursor_scan_ids[spectrum_idx] = precursor_scan_id
        self._mz_values[current_peak_index:peak_stop_idx] = mz_array
        self._intensity_values[current_peak_index:peak_stop_idx] = intensity_array

        # 提取 RT 值
        self.rt_values[spectrum_idx] = rt

        # TODO: 应该还有个 mobility

        """ DIA 循环相关属性 """
        # TODO: 确定 DIA 循环 ，暂时没用没有写
        # self._determine_dia_cycle()

        """ 索引和边界信息 """
        # 创建从 scan_id 到 spec_idx 的映射
        self._scan_id_to_index[scan_id] = spectrum_idx
        # 提取峰索引
        self._peak_start_idx_list[spectrum_idx] = current_peak_index
        self._peak_stop_idx_list[spectrum_idx] = peak_stop_idx

        # 提取这个谱图 mz 范围
        self._precursor_lower_mz[spectrum_idx] = isolation_lower
        self._precursor_upper_mz[spectrum_idx] = isolation_upper

    def _load_from_mzml(
        self,
        mzml_file_path: None | str = None
    ):
        """从 mzML 文件加载数据"""
        logging.info(f"Loading DIA data from {mzml_file_path} ...")

        # 第一遍：统计数据量
        total_spectra = 0
        total_peaks = 0
        with mzml.read(mzml_file_path) as reader:
            for spectrum in reader:
                total_spectra += 1
                total_peaks += len(spectrum['m/z array'])

        logging.info(f"{mzml_file_path} Total spectra: {
                     total_spectra}, total peaks: {total_peaks}")

        # 预先分配数组
        self._preallocate_arrays(total_spectra=total_spectra,
                                 total_peaks=total_peaks)

        # 第二遍：填充数据
        current_spectrum_idx = 0
        current_peak_idx = 0
        # 开始处理信息
        with mzml.read(mzml_file_path) as reader:
            for spectrum in reader:

                self._process_single_spectrum(
                    spectrum, current_spectrum_idx, current_peak_idx)

                # 更新索引
                num_peaks = len(spectrum['m/z array'])
                current_peak_idx += num_peaks
                current_spectrum_idx += 1

        """ mz 范围信息 """
        # 计算 m/z 范围
        if np.all(np.isnan(self._precursor_upper_mz)):
            self._max_mz_value = np.float32(np.nan)
        else:
            self._max_mz_value = np.float32(
                np.nanmax(self._precursor_upper_mz))

        if np.all(np.isnan(self._precursor_lower_mz)):
            self._min_mz_value = np.float32(np.nan)
        else:
            self._min_mz_value = np.float32(
                np.nanmin(self._precursor_lower_mz))

        self.ms1_indexs = np.where(
            self.precursor_scan_ids == -1)[0].astype(np.int32)
        self.ms1_indexs_rt = self.rt_values[self.ms1_indexs].copy()

        # 设置帧索引
        self.frame_max_index = len(self.rt_values) - 1

        self.ms2_indexs = np.where(
            self.precursor_scan_ids != -1)[0].astype(np.int32)
        self.ms2_indexs_rt = self.rt_values[self.ms2_indexs].copy()

        # 更新窗口信息
        if self._precursor_lower_mz is not None:
            self._cycle_left_precursor = deduplicate_with_tolerance(
                self._precursor_lower_mz,
                tolerance=0.1
            )

    def check_in_raw(self, precursor_mz) -> bool:
        """ 检查这个 mz 是否在当前 raw 中"""
        if (precursor_mz <= self._max_mz_value + 0.1
                and precursor_mz >= self._min_mz_value - 0.1):
            return True

        logging.warn("没有找到任何匹配ms2 窗口，可能是重标超出当前 raw 的范围了")
        logging.warn(f"{self._max_mz_value} {self._min_mz_value} precursor_mz: {precursor_mz}  left: {
            self._cycle_left_precursor}")
        return False

    def check_in_same_ms2(self, p1, p2) -> bool:
        """ 检查这两个是否在同一个 ms2 中"""

        idx1 = np.searchsorted(self._cycle_left_precursor, p1)
        idx2 = np.searchsorted(self._cycle_left_precursor, p2)

        return idx1 == idx2

    def get_window_info(self, precursor_mz: float) -> dict:
        """获取包含该 precursor_mz 的 DIA 窗口信息。
        返回 {
            "width": 窗口宽度Da,
            "centering": 前体在窗口中的相对位置 0-1,
            "lower": 窗口下边界 m/z (NaN if not found),
            "upper": 窗口上边界 m/z (NaN if not found),
        }
        """
        default = {"width": 0.0, "centering": 0.5,
                   "lower": float("nan"), "upper": float("nan")}
        if (self._precursor_lower_mz is None or
                self._precursor_upper_mz is None or
                self.ms2_indexs is None or
                len(self.ms2_indexs) == 0):
            return default

        cycle_len = (len(self._cycle_left_precursor)
                     if self._cycle_left_precursor is not None
                     else len(self.ms2_indexs))
        search_range = min(len(self.ms2_indexs), max(cycle_len, 50))

        for i in range(search_range):
            gidx = self.ms2_indexs[i]
            lower = self._precursor_lower_mz[gidx]
            upper = self._precursor_upper_mz[gidx]
            if np.isnan(lower) or np.isnan(upper):
                continue
            if lower - 0.1 <= precursor_mz <= upper + 0.1:
                width = float(upper - lower)
                centering = (float(precursor_mz - lower) / width
                             if width > 0 else 0.5)
                return {"width": width, "centering": centering,
                        "lower": float(lower), "upper": float(upper)}

        return default

    def _check_is_ms1(self, index: int) -> bool:
        """ 检查这个下标对应的谱图是不是一个ms1"""
        if index < 0 or index >= len(self.precursor_scan_ids):
            raise IndexError("Spectrum index out of range")

        if self.precursor_scan_ids[index] == -1:
            return True
        return False

    def get_spectrum_by_index(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """ 根据自己编码的index 返回谱图信息 """
        if index < 0 or index >= len(self.rt_values):
            raise IndexError("Spectrum index out of range")

        start_idx = self._peak_start_idx_list[index]
        stop_idx = self._peak_stop_idx_list[index]

        mz = self._mz_values[start_idx:stop_idx]
        intensity = self._intensity_values[start_idx:stop_idx]

        return mz, intensity

    def _ms2_cycle_idx(self, global_ms2_idx: int) -> int:
        """Return the cycle index (= position in ms1_indexs) that owns this MS2.

        DIA cycle = one MS1 followed by N MS2. The owning MS1 is identified by
        precursor_scan_ids[global_ms2_idx]. Return -1 if the owning MS1 isn't
        found in ms1_indexs (defensive; shouldn't happen on well-formed data).
        """
        if (self.precursor_scan_ids is None or
                self._scan_id_to_index is None or
                self.ms1_indexs is None):
            return -1
        ms1_scan_id = int(self.precursor_scan_ids[global_ms2_idx])
        if ms1_scan_id < 0:
            return -1
        ms1_global_idx = int(self._scan_id_to_index[ms1_scan_id])
        pos = int(np.searchsorted(self.ms1_indexs, ms1_global_idx))
        if (pos < len(self.ms1_indexs) and
                int(self.ms1_indexs[pos]) == ms1_global_idx):
            return pos
        return -1

    def get_spectrum(self, scan_id: int) -> tuple[np.ndarray, np.ndarray]:
        """获取指定索引的谱图数据"""
        if scan_id < 0 or scan_id >= len(self._scan_id_to_index):
            raise IndexError("Spectrum index out of range")

        index = self._scan_id_to_index[scan_id]

        return self.get_spectrum_by_index(index)

    def get_ms1_spectrum_by_ms1_index(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """ 根据提供的ms2的index获取ms1的谱图信息 """
        if index < 0 or index >= len(self.precursor_scan_ids):
            raise IndexError("Spectrum index out of range")

        ms1_scan_id = self.precursor_scan_ids[index]

        return self.get_spectrum(ms1_scan_id)

    def get_spectrum_by_rt(
        self, rt: np.float32, precurso_mz: np.float32
    ) -> tuple[np.ndarray, np.ndarray]:
        """ 根据rt 获得这个谱图信息 """

        # NOTE: 假设 RT 数组是根据时间递增的
        # 那么这里使用二分来查找对应的 index
        idx = np.searchsorted(self.rt_values, rt)

        logging.info(f"idx: {idx}")
        return self.get_spectrum_by_index(idx)

    def xic_ms2_peaks_extract(
        self,
        rt: np.float32,
        xic_cycle_window: int,
        precursor_mz: np.float32,
        ions_mass: np.float32,
        mass_tol_ppm: np.float32,
    ) -> tuple[np.ndarray, np.float32]:
        """
        提取 XIC：从最接近 rt 的、且 precursor_mz 在隔离窗口内的 MS2 谱图开始，
        向左/右各扩展 xic_cycle_window 个「有效」MS2 谱图（即窗口包含 precursor_mz）。
        """
        if self.ms2_indexs is None or len(self.ms2_indexs) == 0:
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("cycle_idx", "i4")]
            return np.array([], dtype=dtype), 0.0

        protonmass = 1.00727646677
        ans = []
        total_intensity = 0.0

        # Step 1: 找到 _ms2_rt_values 中最接近 rt 的位置
        pos = np.searchsorted(self.ms2_indexs_rt, rt)

        # 候选中心点
        candidates = []
        for i in range(1, 6):
            if pos - i >= 0:
                candidates.append(pos - i)
        for i in range(0, 6):
            if pos + i < len(self.ms2_indexs_rt):
                candidates.append(pos + i)

        center_idx = None
        min_diff = float('inf')
        for i in candidates:
            global_idx = self.ms2_indexs[i]
            lower = self._precursor_lower_mz[global_idx] - 0.1
            upper = self._precursor_upper_mz[global_idx] + 0.1

            if np.isnan(lower) or np.isnan(upper):
                continue
            if lower <= precursor_mz <= upper:
                diff = abs(self.ms2_indexs_rt[i] - rt)
                if diff < min_diff:
                    min_diff = diff
                    center_idx = i  # 在 _ms2_indices 中的位置

        if center_idx is None:
            # 没有找到任何窗口匹配的 MS2 谱图
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("cycle_idx", "i4")]
            logging.warn("没有找到任何匹配ms2 窗口，可能是重标超出当前 raw 的范围了")
            logging.warn(f"precursor_mz: {precursor_mz}  left: {
                         self._cycle_left_precursor}")

            for i in candidates:
                gidx = self.ms2_indexs[i]
                lower = self._precursor_lower_mz[gidx]
                upper = self._precursor_upper_mz[gidx]
                logging.warn(f"precursor_mz: {precursor_mz} candidate idx={i}, global={gidx}, rt={
                             self.ms2_indexs_rt[i]:.3f}, window=[{lower:.3f}, {upper:.3f})")
            return np.array([], dtype=dtype), 0.0

        # Step 2: 向左收集 xic_cycle_window 个有效谱图
        left_list = []
        i = center_idx - 1
        while i >= 0 and len(left_list) < xic_cycle_window:
            global_idx = self.ms2_indexs[i]
            lower = self._precursor_lower_mz[global_idx]
            upper = self._precursor_upper_mz[global_idx]
            if (not (np.isnan(lower) or np.isnan(upper)) and
                    lower <= precursor_mz <= upper):
                left_list.append(global_idx)
            i -= 1

        # Step 3: 向右收集 xic_cycle_window 个有效谱图
        right_list = []
        i = center_idx + 1
        while (i < len(self.ms2_indexs) and
               len(right_list) < xic_cycle_window):
            global_idx = self.ms2_indexs[i]
            lower = self._precursor_lower_mz[global_idx]
            upper = self._precursor_upper_mz[global_idx]
            if (not (np.isnan(lower) or np.isnan(upper))
                    and lower <= precursor_mz <= upper):
                right_list.append(global_idx)
            i += 1

        # Step 4: 合并（左 + 中心 + 右）
        selected_global_indices = left_list[::-1] + \
            [self.ms2_indexs[center_idx]] + right_list

        # logging.info(selected_global_indices)

        # Step 5: 处理每个谱图
        for global_idx in selected_global_indices:
            mz_arr, intensity_arr = self.get_spectrum_by_index(global_idx)
            total_intensity += np.sum(intensity_arr)

            ppm_error = 0.0
            match_intensity = 0.0

            for charge in range(1, 3):
                theo_mz = (ions_mass + charge * protonmass) / charge
                tot_ppm_error, tot_match_intensity = match_peak_ppm(
                    mz_arr, intensity_arr, theo_mz, mass_tol_ppm
                )
                if not np.isnan(tot_ppm_error):
                    ppm_error += tot_ppm_error
                match_intensity += tot_match_intensity

            ans.append({
                "rt": self.rt_values[global_idx],
                "ppm_error": ppm_error,
                "intensity": match_intensity,
                "cycle_idx": self._ms2_cycle_idx(int(global_idx)),
            })

        dtype = [("rt", "f8"), ("ppm_error", "f8"),
                 ("intensity", "f8"), ("cycle_idx", "i4")]
        arr = np.array([tuple(d.values()) for d in ans], dtype=dtype)

        return arr, total_intensity

    def find_near_ms1_idx(self, rt: np.float32):
        """ 找到那个离这个 rt 更加接近 """
        idx = np.searchsorted(self.ms1_indexs_rt, rt)

        if idx == 0:
            return 0
        if idx == len(self.ms1_indexs_rt):
            return idx - 1

        left = self.ms1_indexs_rt[idx - 1]
        right = self.ms1_indexs_rt[idx]

        if abs(rt - left) <= abs(rt - right):
            return idx - 1
        else:
            return idx

    def xic_peaks_extreact(
        self,
        rt: np.float32, xic_cycle_window: int,
        precursor_mz: np.float32,
        mass_tol_ppm: np.float32,
    ) -> np.ndarray:
        """ 过滤出这些保留时间内所有的ms1谱图，然后返回peaks  """

        ans = []

        # 先寻找的起始的 index
        mid_rt_index = self.find_near_ms1_idx(rt)

        start_index = max(0, mid_rt_index - xic_cycle_window)
        end_index = min(len(self.ms1_indexs),
                        mid_rt_index + xic_cycle_window + 1)

        # 遍历所有 index, 并记录 cycle_idx (= position in ms1_indexs)
        for local_pos, index in enumerate(
                self.ms1_indexs[start_index:end_index]):
            cycle_idx = start_index + local_pos

            # 当是 ms1 谱图的时候，取出这个precursor_mz 对应的信息
            (mz_arr, intensity_arr) = self.get_spectrum_by_index(index)

            (ppm_error, match_intensity) = match_peak_ppm(
                mz_arr, intensity_arr, precursor_mz, mass_tol_ppm)

            ans.append(
                {"rt": self.rt_values[index],
                 "ppm_error": ppm_error,
                 "intensity": match_intensity,
                 "cycle_idx": cycle_idx})

        dtype = [("rt", "f8"), ("ppm_error", "f8"),
                 ("intensity", "f8"), ("cycle_idx", "i4")]

        # 把 list[dict] 转成结构化 ndarray
        arr = np.array([tuple(d.values()) for d in ans], dtype=dtype)

        return arr
