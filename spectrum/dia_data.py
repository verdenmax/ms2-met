
import re
import logging
import numpy as np
import pandas as pd

from pyteomics import mzml
from spectrum.spectrum_utils import match_peak_ppm


DEFAULT_VALUE_NO_MOBILITY = 1e-6


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
        # 这个ms2 index 对应的 ms1 信息
        self.precursor_scan_ids: np.ndarray[tuple[int],
                                            np.dtype[np.int32]] = None
        self._mz_values: np.ndarray[tuple[int],
                                    np.dtype[np.float32]] | None = None
        self.rt_values: np.ndarray[tuple[int],
                                   np.dtype[np.float32]] | None = None
        self._intensity_values: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)
        self.mobility_values: (
            np.ndarray[tuple[int], np.dtype[np.float32]]) = np.array(
            [DEFAULT_VALUE_NO_MOBILITY, 0], dtype=np.float32
        )

        """ DIA 循环相关属性 """
        # DIA 循环定义数组
        self._precursor_max_mz_value: np.float32 = None
        self._precursor_min_mz_value: np.float32 = None
        self._cycle_left_precursor: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)

        """ mz 范围信息 """
        self._max_mz_value: np.float32 | None = None
        self._min_mz_value: np.float32 | None = None
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
        self._max_mz_value = np.float32(np.max(self._mz_values))
        self._min_mz_value = np.float32(np.min(self._mz_values))

        # 设置帧索引
        self.frame_max_index = len(self.rt_values) - 1

        self._determine_dia_cycle()

    def _preprocess_data(self):
        """预处理数据，填充所有属性"""

        """ 数据特征标识 """
        # 检查是否有 MS1 数据
        self.has_ms1 = (self.spectrum_df['ms_level'] == 1).any()

        if not self.has_ms1:
            logging.warn("No MS1 spectra found in the file")

        """
        记录的原始数据关键数组, mz_value、rt_value、intensity_value、mobility_values。
        """
        # 提取 m/z 和强度值
        self.precursor_scan_ids = (
            self.spectrum_df['precursor_scan_id'].values.astype(np.int64))
        self._mz_values = self.peak_df['mz'].values.astype(np.float32)
        self._intensity_values = self.peak_df['intensity'].values.astype(
            np.float32)

        # 提取 RT 值
        self.rt_values = self.spectrum_df['rt'].values.astype(np.float32)

        # TODO: 应该还有个 mobility

        """ DIA 循环相关属性 """
        # TODO: 确定 DIA 循环 ，暂时没用没有写
        # self._determine_dia_cycle()

        """ mz 范围信息 """
        # 计算 m/z 范围
        self._max_mz_value = np.float32(self.peak_df['mz'].max())
        self._min_mz_value = np.float32(self.peak_df['mz'].min())

        # 计算四极杆 m/z 范围 (仅 MS2)
        ms2_spectra = self.spectrum_df[self.spectrum_df['ms_level'] == 2]
        if not ms2_spectra.empty:
            self._quad_max_mz_value = np.array(
                [ms2_spectra['isolation_upper_mz'].max()], dtype=np.float32)
            self._quad_min_mz_value = np.array(
                [ms2_spectra['isolation_lower_mz'].min()], dtype=np.float32)
        else:
            self._quad_max_mz_value = np.array([0], dtype=np.float32)
            self._quad_min_mz_value = np.array([0], dtype=np.float32)

        """ 索引和边界信息 """
        # 创建从 scan_id 到 spec_idx 的映射
        self._scan_id_to_index = (
            self.spectrum_df.set_index('scan_id')['spec_idx']
            .reindex(range(self.spectrum_df['scan_id'].max() + 1))
            .fillna(0)  # 或者用其他默认值填充缺失的 scan_id
            .values.astype(np.int64)
        )
        # 提取峰索引
        self._peak_start_idx_list = (
            self.spectrum_df['peak_start_idx'].values.astype(np.int64))
        self._peak_stop_idx_list = (
            self.spectrum_df['peak_stop_idx'].values.astype(np.int64))

        # 提取这个谱图 mz 范围
        self._precursor_lower_mz = (
            self.spectrum_df['isolation_lower_mz'].values.astype(np.float32))
        self._precursor_upper_mz = (
            self.spectrum_df['isolation_upper_mz'].values.astype(np.float32))

        # 设置帧索引
        self.frame_max_index = len(self.rt_values) - 1

        # clean
        del self.spectrum_df
        del self.peak_df

    def _determine_dia_cycle(self):
        """确定 DIA 循环结构"""
        # 简化的 DIA 循环检测

        ms1_indices = [index for index, scan_id in enumerate(
            self.precursor_scan_ids) if scan_id == -1]

        if len(ms1_indices) == 0:
            logging.warn("Cannot determine DIA cycle")
            return
        elif len(ms1_indices) == 1:
            # 设置为最后一个
            ms1_indices.append(len(self.precursor_scan_ids + 1))

        self._cycle_left_precursor = (
            self._precursor_lower_mz[ms1_indices[0]+1:ms1_indices[1]])

        # 给出precursor最大值和最小值
        self._precursor_min_mz_value = (np.min(
            self._precursor_lower_mz[ms1_indices[0]+1:ms1_indices[1]]))
        self._precursor_max_mz_value = np.max(
            self._precursor_upper_mz[ms1_indices[0]+1:ms1_indices[1]])

    def check_in_same_ms2(self, p1, p2) -> bool:
        """ 检查这两个是否在同一个 ms2 中"""

        idx1 = np.searchsorted(self._cycle_left_precursor, p1)
        idx2 = np.searchsorted(self._cycle_left_precursor, p2)

        return idx1 == idx2

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
        rt_start: np.float32, rt_stop: np.float32,
        precursor_mz: np.float32,
        ions_mass: np.float32,
        mass_tol_ppm: np.float32,
    ) -> np.ndarray:
        """ 过滤出这些保留时间内所有的ms2谱图，然后返回peaks  """
        ans = []
        protonmass = 1.00727646677  # mass.calculate_mass(formula='H+')

        # 先寻找的起始的 index
        start_idx = np.searchsorted(self.rt_values, rt_start)
        end_idx = np.searchsorted(self.rt_values, rt_stop)

        # 遍历所有 index
        for index in range(start_idx, end_idx):
            # NOTE:  这里现在是如果是 ms1 就 continue
            if self._check_is_ms1(index):
                continue

            # NOTE: 加上如果当前母离子范围不对，也 continue
            if (precursor_mz > self._precursor_upper_mz[index] or
                    precursor_mz < self._precursor_lower_mz[index]):
                continue

            # 当是 ms2 谱图的时候，取出这个precursor_mz 对应的信息
            (mz_arr, intensity_arr) = self.get_spectrum_by_index(index)

            ppm_error = np.nan
            match_intensity = 0

            # NOTE: 这里最好将多个 电荷的这个累计起来
            for charge in range(1, 3):
                theo_mz = (ions_mass + charge * protonmass) / charge

                # 计算出结果之后
                (tot_ppm_error, tot_match_intensity) = match_peak_ppm(
                    mz_arr, intensity_arr, theo_mz, mass_tol_ppm)

                # 累计结果
                if not tot_ppm_error == np.nan:
                    ppm_error += tot_ppm_error
                match_intensity += tot_match_intensity

            ans.append(
                {"rt": self.rt_values[index],
                 "pmm_error": ppm_error,
                 "intensity": match_intensity})

        dtype = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8")]

        # 把 list[dict] 转成结构化 ndarray
        arr = np.array([tuple(d.values()) for d in ans], dtype=dtype)

        return arr

    def xic_peaks_extreact(
        self,
        rt_start: np.float32, rt_stop: np.float32,
        precursor_mz: np.float32,
        mass_tol_ppm: np.float32,
    ) -> np.ndarray:
        """ 过滤出这些保留时间内所有的ms1谱图，然后返回peaks  """

        ans = []

        # 先寻找的起始的 index
        start_idx = np.searchsorted(self.rt_values, rt_start)
        end_idx = np.searchsorted(self.rt_values, rt_stop)

        # 遍历所有 index
        for index in range(start_idx, end_idx):

            if not self._check_is_ms1(index):
                continue

            # 当是 ms1 谱图的时候，取出这个precursor_mz 对应的信息
            (mz_arr, intensity_arr) = self.get_spectrum_by_index(index)

            (pmm_error, match_intensity) = match_peak_ppm(
                mz_arr, intensity_arr, precursor_mz, mass_tol_ppm)

            ans.append(
                {"rt": self.rt_values[index],
                 "pmm_error": pmm_error,
                 "intensity": match_intensity})

        dtype = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8")]

        # 把 list[dict] 转成结构化 ndarray
        arr = np.array([tuple(d.values()) for d in ans], dtype=dtype)

        return arr
