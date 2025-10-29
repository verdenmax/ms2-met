
import re
import logging
import numpy as np
import pandas as pd

from pyteomics import mzml


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
        self.cycle: (
            np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]] | None
        ) = None
        self._cycle_start: int | None = None
        self._cycle_length: int | None = None
        self._precursor_cycle_max_index: int | None = None

        """ mz 范围信息 """
        self._max_mz_value: np.float32 | None = None
        self._min_mz_value: np.float32 | None = None
        self._quad_max_mz_value: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)
        self._quad_min_mz_value: (
            np.ndarray[tuple[int], np.dtype[np.float32]] | None) = (None)

        """ 索引和边界信息 """
        self._scan_id_to_index: (
            np.ndarray[tuple[int], np.dtype[np.int64]] | None) = (None)
        self._peak_start_idx_list: (
            np.ndarray[tuple[int], np.dtype[np.int64]] | None) = (None)
        self._peak_stop_idx_list: (
            np.ndarray[tuple[int], np.dtype[np.int64]] | None) = (None)
        self._zeroth_frame: int = 0
        self._scan_max_index: int = 1
        self.frame_max_index: int | None = None

    def _get_retention_time(self, spectrum) -> float:
        """从谱图中提取保留时间（转换为秒）"""

        if 'scanList' in spectrum:
            scan = spectrum['scanList']['scan'][0]
            if 'scan start time' in scan:
                rt = scan['scan start time']
                # 检查单位并转换为秒
                if isinstance(rt, dict):
                    unit = rt.get('unitName', 'minute')
                    value = rt.get('value', 0)
                    if unit == 'minute':
                        return value * 60
                    elif unit == 'second':
                        return value
                else:
                    # 假设单位是分钟
                    return rt * 60
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

    def _load_from_mzml(
        self,
        mzml_file_path: None | str = None
    ):
        """从 mzML 文件加载数据"""
        logging.info(f"Loading DIA data from {mzml_file_path} ...")

        spectra_data = []
        peak_data = []
        current_peak_index = 0

        # 开始读取信息
        with mzml.read(mzml_file_path) as reader:
            for spectrum_idx, spectrum in enumerate(reader):
                # 获取保留时间 (转换为秒)
                rt = self._get_retention_time(spectrum)

                # 获取质谱的scan id，不需要使用 spectrum_idx
                scan_id = self._extract_scan_number(spectrum['id'])

                # 获取spec title
                spec_title = spectrum.get('spectrum title', None).split()[0]

                # 获取 MS 级别
                ms_level = spectrum.get('ms level', 1)

                # 获取前体信息 (对于 MS2)
                precursor_mz = None
                precursor_charge = None
                precursor_intensity = None
                isolation_lower = None
                isolation_upper = None

                if ms_level > 1 and 'precursorList' in spectrum:
                    precursors = spectrum['precursorList']['precursor']

                    if precursors:
                        precursor = precursors[0]
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
                spectrum_info = {
                    'spec_idx': spectrum_idx,
                    'scan_id': scan_id,
                    'rt': rt,
                    'spec_title': spec_title,
                    'ms_level': ms_level,
                    'precursor_mz': precursor_mz,
                    'precursor_charge': precursor_charge,
                    'precursor_intensity': precursor_intensity,
                    'isolation_lower_mz': isolation_lower,
                    'isolation_upper_mz': isolation_upper,
                    'peak_start_idx': current_peak_index,
                    'peak_stop_idx': current_peak_index + len(mz_array)
                }

                spectra_data.append(spectrum_info)

                # 记录峰数据
                for mz, intensity in zip(mz_array, intensity_array):
                    peak_data.append({
                        'spec_idx': spectrum_idx,
                        'mz': mz,
                        'intensity': intensity
                    })

                current_peak_index += len(mz_array)

        # 创建 DataFrame
        self.spectrum_df = pd.DataFrame(spectra_data)
        self.peak_df = pd.DataFrame(peak_data)

        del spectra_data
        del peak_data

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

        # 设置帧索引
        self.frame_max_index = len(self.rt_values) - 1

        # clean
        del self.spectrum_df
        del self.peak_df

    def _determine_dia_cycle(self):
        """确定 DIA 循环结构"""
        # 简化的 DIA 循环检测
        # 在实际应用中，你可能需要更复杂的逻辑来检测循环模式

        ms1_indices = self.spectrum_df[self.spectrum_df['ms_level'] == 1].index
        ms2_indices = self.spectrum_df[self.spectrum_df['ms_level'] == 2].index

        if len(ms1_indices) == 0 or len(ms2_indices) == 0:
            logging.warn("Cannot determine DIA cycle")
            return

        # 假设第一个 MS1 是循环开始
        self._cycle_start = ms1_indices[0]

        # 计算平均循环长度 (MS1 之间的间隔)
        if len(ms1_indices) > 1:
            cycle_lengths = np.diff(ms1_indices)
            self._cycle_length = int(np.median(cycle_lengths))
        else:
            # 如果只有一个 MS1，使用 MS2 的数量作为循环长度
            self._cycle_length = len(ms2_indices)

        # 构建 cycle 数组 (简化版本)
        # 在实际应用中，你需要根据实际的隔离窗口信息来构建
        num_cycles = len(ms1_indices)
        num_windows = self._cycle_length - 1  # 假设每个循环有1个MS1和多个MS2

        self.cycle = np.zeros((4, num_windows, num_cycles), dtype=np.float64)

        for cycle_idx, ms1_idx in enumerate(ms1_indices):
            if cycle_idx < num_cycles:
                # 获取这个循环中的 MS2 谱图
                cycle_end = ms1_idx + \
                    self._cycle_length if cycle_idx < len(
                        ms1_indices) - 1 else len(self.spectrum_df)
                cycle_ms2 = self.spectrum_df.iloc[ms1_idx + 1:cycle_end]
                cycle_ms2 = cycle_ms2[cycle_ms2['ms_level'] == 2]

                for window_idx, (_, ms2_spec) in enumerate(cycle_ms2.iterrows()):
                    if window_idx < num_windows:
                        self.cycle[0, window_idx,
                                   cycle_idx] = ms1_idx  # MS1 index
                        self.cycle[1, window_idx,
                                   cycle_idx] = ms2_spec.name  # MS2 index
                        self.cycle[2, window_idx,
                                   cycle_idx] = ms2_spec['isolation_lower_mz'] or 0
                        self.cycle[3, window_idx,
                                   cycle_idx] = ms2_spec['isolation_upper_mz'] or 0

        self._precursor_cycle_max_index = len(
            self.rt_values) // (self._cycle_length if self._cycle_length else 1)

    def get_spectrum(self, scan_id: int) -> tuple[np.ndarray, np.ndarray]:
        """获取指定索引的谱图数据"""
        if scan_id < 0 or scan_id >= len(self._scan_id_to_index):
            raise IndexError("Spectrum index out of range")

        index = self._scan_id_to_index[scan_id]

        start_idx = self._peak_start_idx_list[index]
        stop_idx = self._peak_stop_idx_list[index]

        mz = self._mz_values[start_idx:stop_idx]
        intensity = self._intensity_values[start_idx:stop_idx]

        return mz, intensity
