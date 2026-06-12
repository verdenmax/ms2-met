
import os
import re
import logging
import tempfile
import numpy as np
import pandas as pd

from pyteomics import mzml
from spectrum.spectrum_utils import match_peak_ppm, centroid_spectrum


DEFAULT_VALUE_NO_MOBILITY = 1e-6

# Centroid params default (P0-3, Silent-C3, 2026-06-03 audit).
# Single source of truth shared by DIAData.__init__, DataManager
# config injection, and flow_utils cache validation.
DEFAULT_CENTROID_ENABLED: bool = True
DEFAULT_CENTROID_REL_THRESHOLD: float = 1e-3


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


def _is_already_centroid(spectrum) -> bool:
    """Return True if the pyteomics spectrum dict carries the MS controlled-
    vocabulary term `MS:1000127 centroid spectrum`.

    pyteomics flattens cv terms into dict keys with the term name as the key
    (and the value as the term's value, often empty string). Presence of the
    key alone is sufficient.
    """
    return 'centroid spectrum' in spectrum


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

    # NOTE: `_format_version` is intentionally NOT loaded here — it's
    # consumed by `DIAData._check_format_version` before this function
    # runs. Do not iterate `data.files` blindly to copy keys onto `obj`.

    # 可选数组
    obj._quad_max_mz_value = data['_quad_max_mz_value'] if '_quad_max_mz_value' in data else None
    obj._quad_min_mz_value = data['_quad_min_mz_value'] if '_quad_min_mz_value' in data else None
    obj._scan_id_to_index = data['_scan_id_to_index'] if '_scan_id_to_index' in data else None
    obj._peak_start_idx_list = data['_peak_start_idx_list'] if '_peak_start_idx_list' in data else None
    obj._peak_stop_idx_list = data['_peak_stop_idx_list'] if '_peak_stop_idx_list' in data else None
    obj._precursor_lower_mz = data['_precursor_lower_mz'] if '_precursor_lower_mz' in data else None
    obj._precursor_upper_mz = data['_precursor_upper_mz'] if '_precursor_upper_mz' in data else None

    # Centroid params (P0-3, added in _format_version=3).
    if '_centroid_enabled' in data:
        obj._centroid_enabled = bool(data['_centroid_enabled'])
    if '_centroid_rel_threshold' in data:
        obj._centroid_rel_threshold = float(data['_centroid_rel_threshold'])


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

        """ 加载时 centroiding 配置 (由 DataManager 从 config 注入；这里给默认值) """
        self._centroid_enabled: bool = DEFAULT_CENTROID_ENABLED
        self._centroid_rel_threshold: float = DEFAULT_CENTROID_REL_THRESHOLD

        # P1-6 (Silent-I3, 2026-06-03 audit): counter incremented on each
        # out-of-window XIC request; per-worker summary logged at batch
        # end in workflows/flow_utils.py.
        self._n_out_of_window_xic: int = 0

        # P1-7 (Silent-I8, 2026-06-03 audit): counter incremented in
        # _load_from_mzml when centroid_spectrum returns empty for a
        # short/all-zero spectrum; summary logged at end of load.
        self._n_centroid_empty: int = 0

    # 在 DIAData 类中添加
    def save_to_file(self, filepath: str, source_path: str | None = None):
        """将所有 NumPy 数组和标量保存到 .npz 文件（原子写）。

        source_path: 源 mzML 路径；提供则把其 mtime/size 写入缓存，供
        validate_cache_params 检测源文件是否被替换/重新生成（缓存失效）。
        """

        data = {
            # 格式版本号 (3 = centroided peaks + embedded centroid params; 见 P0-3)
            '_format_version': np.int32(3),
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

            # Centroid params for cache invalidation (P0-3, Silent-C3, 2026-06-03).
            '_centroid_enabled': np.bool_(self._centroid_enabled),
            '_centroid_rel_threshold': np.float64(self._centroid_rel_threshold),
        }

        # 源文件身份（用于缓存失效检测；源 mzML 被替换/重新生成时重建）
        if source_path is not None and os.path.exists(source_path):
            st = os.stat(source_path)
            data['_source_mtime'] = np.float64(st.st_mtime)
            data['_source_size'] = np.int64(st.st_size)

        # 过滤掉 None 值（np.savez 不支持 None）
        data = {k: v for k, v in data.items() if v is not None}

        # 原子写：先写同目录临时 .npz，再 os.replace；避免崩溃/并发留下截断文件
        out_dir = os.path.dirname(filepath) or "."
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=out_dir)
        os.close(fd)
        try:
            np.savez_compressed(tmp, **data)
            os.replace(tmp, filepath)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        logging.info(f"Saved DIAData to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str, use_mmap: bool = True,
                       expected_centroid_enabled: bool | None = None,
                       expected_centroid_rel_threshold: float | None = None):
        """从 .npz 文件加载 DIAData，支持内存映射（只读）

        Args:
            filepath: npz cache path.
            use_mmap: zero-copy mmap mode.
            expected_centroid_enabled: if provided, reject cache if mismatched
                (P0-3, Silent-C3 in 2026-06-03 deep audit).
            expected_centroid_rel_threshold: if provided, reject cache if
                |delta| > 1e-12.

        Raises:
            ValueError: if _format_version != 3 OR centroid params mismatch
                expected values.
        """
        obj = cls()

        if use_mmap:
            with np.load(filepath, mmap_mode='r') as data:
                cls._check_format_version(filepath, data,
                                          expected_centroid_enabled,
                                          expected_centroid_rel_threshold)
                _load_attrs(obj, data)
        else:
            with np.load(filepath) as data:
                cls._check_format_version(filepath, data,
                                          expected_centroid_enabled,
                                          expected_centroid_rel_threshold)
                _load_attrs(obj, data)

        return obj

    @staticmethod
    def validate_cache_params(filepath: str,
                              expected_centroid_enabled: bool,
                              expected_centroid_rel_threshold: float,
                              expected_source_path: str | None = None) -> None:
        """Lightweight cache validation: open npz with mmap, read ONLY the
        scalars needed for version + centroid (+ optional source-identity)
        checks, then close.

        Avoids materializing multi-GB arrays just to validate metadata
        (P0-3 review I1). Uses `with np.load(...)` to guarantee handle
        closure before any subsequent os.remove (P0-3 review I2 —
        Windows file-handle race).

        Raises:
            ValueError: if _format_version != 3, centroid params mismatch, or
                (when expected_source_path given) the source mzML's size/mtime
                differs from what was cached.
        """
        with np.load(filepath, mmap_mode='r') as data:
            DIAData._check_format_version(
                filepath, data,
                expected_centroid_enabled=expected_centroid_enabled,
                expected_centroid_rel_threshold=expected_centroid_rel_threshold,
                expected_source_path=expected_source_path,
            )

    @staticmethod
    def _check_format_version(filepath: str, data,
                              expected_centroid_enabled: bool | None = None,
                              expected_centroid_rel_threshold: float | None = None,
                              expected_source_path: str | None = None) -> None:
        """Reject npz files without `_format_version=3` or mismatched centroid params.

        Bumped from 2 -> 3 in P0-3 (Silent-C3) to embed centroid params
        in the cache. Caller passes the currently-configured centroid
        params; mismatch raises (forcing rebuild).
        """
        if '_format_version' not in data:
            raise ValueError(
                f"npz 缓存 {filepath} 没有 _format_version 字段——这是 "
                f"旧版本（profile peaks）生成的缓存。请删除该文件后重新"
                f"运行以生成 centroided 缓存。"
            )
        version = int(data['_format_version'])
        if version != 3:
            raise ValueError(
                f"npz 缓存 {filepath} 的 _format_version={version}，"
                f"当前代码只支持 version=3。请删除该文件后重新运行。"
            )
        if expected_centroid_enabled is not None:
            stored_enabled = bool(data['_centroid_enabled']) \
                if '_centroid_enabled' in data else None
            if stored_enabled != expected_centroid_enabled:
                raise ValueError(
                    f"npz 缓存 {filepath} 的 _centroid_enabled={stored_enabled}, "
                    f"配置要求 {expected_centroid_enabled}。请删除该文件后重新运行。"
                )
        if expected_centroid_rel_threshold is not None:
            stored_threshold = float(data['_centroid_rel_threshold']) \
                if '_centroid_rel_threshold' in data else None
            if (stored_threshold is None
                    or abs(stored_threshold - expected_centroid_rel_threshold) > 1e-12):
                raise ValueError(
                    f"npz 缓存 {filepath} 的 _centroid_rel_threshold={stored_threshold}, "
                    f"配置要求 {expected_centroid_rel_threshold}。请删除该文件后重新运行。"
                )
        # 源文件身份校验：仅当调用方给出期望源路径、缓存内含身份字段、且源文件存在
        # （旧缓存无 _source_size 字段时跳过，保持向后兼容命中）
        if (expected_source_path is not None and '_source_size' in data
                and os.path.exists(expected_source_path)):
            st = os.stat(expected_source_path)
            stored_size = int(data['_source_size'])
            stored_mtime = (float(data['_source_mtime'])
                            if '_source_mtime' in data else None)
            if stored_size != st.st_size or (
                    stored_mtime is not None
                    and abs(stored_mtime - st.st_mtime) > 1e-6):
                raise ValueError(
                    f"npz 缓存 {filepath} 的源文件 {expected_source_path} 已变化"
                    f"（size/mtime 不符），需重建。")

    def _get_retention_time(self, spectrum) -> float:
        """Return retention time in MINUTES (canonical pipeline unit).

        pyteomics attaches a `unit_info` attribute to the scalar (e.g.,
        'minute' from MS CV UO:0000031, 'second' from UO:0000010).
        If unit is 'second', convert to minutes. Plain floats without
        unit_info are assumed to be minutes (back-compat).

        Returns 0.0 if no scan-start-time field is present.

        (P2-2, Units-I3, 2026-06-03 deep audit.)
        """
        if 'scanList' in spectrum:
            scan = spectrum['scanList']['scan'][0]
            if 'scan start time' in scan:
                rt = scan['scan start time']
                unit = getattr(rt, 'unit_info', None)
                value = float(rt)
                if unit == 'second':
                    return value / 60.0
                if unit is None or unit == 'minute':
                    return value
                raise ValueError(
                    f"Unsupported RT unit {unit!r}; expected 'minute' or "
                    f"'second'. (P2-2, Units-I3)")
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

    def _preallocate_arrays(self, total_spectra: int, max_scan_id: int):
        """预先分配按谱图数定长的数组。

        Peak 数组 (_mz_values / _intensity_values) 不再预分配，由
        _load_from_mzml 通过 chunk + concat 构建。

        Args:
            total_spectra: 谱图总数，决定按 spectrum_idx 索引的数组长度。
            max_scan_id: 当前 mzML 中所有谱图 scan_id 的最大值。决定
                _scan_id_to_index 反查表的长度（max_scan_id + 1）。
                通常 max_scan_id == total_spectra - 1（scan_id 稠密）；
                pParse / ProteoWizard 过滤后剩余谱图保留原始 scan_num
                时，max_scan_id 可远大于 total_spectra。
        """
        # 谱图信息数组
        self.precursor_scan_ids = np.zeros(total_spectra, dtype=np.int64)
        self.rt_values = np.zeros(total_spectra, dtype=np.float32)
        self._peak_start_idx_list = np.zeros(total_spectra, dtype=np.int64)
        self._peak_stop_idx_list = np.zeros(total_spectra, dtype=np.int64)
        self._precursor_lower_mz = np.zeros(total_spectra, dtype=np.float32)
        self._precursor_upper_mz = np.zeros(total_spectra, dtype=np.float32)

        # scan_id 反查表: 按 max(scan_id) + 1 sizing, 避免 IndexError。
        # max_scan_id == -1 (空 mzML 或所有 id 无 scan 号) 时退化为 size 1。
        scan_id_table_size = max(max_scan_id + 1, 1)
        self._scan_id_to_index = np.zeros(scan_id_table_size, dtype=np.int64)

    def _record_spectrum(
        self, spectrum_idx: int, current_peak_index: int, *,
        scan_id: int, rt: float, precursor_scan_id: int,
        isolation_lower, isolation_upper,
        mz_array: np.ndarray, intensity_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把一张谱图的归一化字段写入按谱图定长的数组（格式无关）。

        isolation_lower/upper 为 None 时（MS1）numpy 自动存为 NaN。
        返回 (mz_array, intensity_array) 供调用方累积 chunk。
        """
        peak_stop_idx = current_peak_index + len(mz_array)
        self.precursor_scan_ids[spectrum_idx] = precursor_scan_id
        self.rt_values[spectrum_idx] = rt
        self._scan_id_to_index[scan_id] = spectrum_idx
        self._peak_start_idx_list[spectrum_idx] = current_peak_index
        self._peak_stop_idx_list[spectrum_idx] = peak_stop_idx
        self._precursor_lower_mz[spectrum_idx] = isolation_lower
        self._precursor_upper_mz[spectrum_idx] = isolation_upper
        return mz_array, intensity_array

    def _process_single_spectrum(
        self, spectrum,
        spectrum_idx: int, current_peak_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ 处理单个的谱图，将其中信息记录起来 """

        # 获取保留时间 (转换为秒)
        rt = self._get_retention_time(spectrum)

        # 获取质谱的scan id，不需要使用 spectrum_idx
        scan_id = self._extract_scan_number(spectrum['id'])

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

        # On-load centroiding (spec 2026-06-01-mzml-centroiding-on-load §5.2).
        # Skip if input already carries the centroid cv term, or if disabled.
        if self._centroid_enabled and not _is_already_centroid(spectrum):
            mz_array, intensity_array = centroid_spectrum(
                mz_array, intensity_array,
                rel_threshold=self._centroid_rel_threshold,
            )
            # P1-7 (Silent-I8, 2026-06-03 audit): count empty-return so
            # we can log a summary at end of load.
            if len(mz_array) == 0:
                self._n_centroid_empty += 1

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

        改造（spec 2026-06-01）：mz/intensity 不再写入预分配数组，而是
        作为 chunk 返回给 _load_from_mzml 累积后 concat。
        """
        return self._record_spectrum(
            spectrum_idx, current_peak_index,
            scan_id=scan_id, rt=rt,
            precursor_scan_id=precursor_scan_id,
            isolation_lower=isolation_lower,
            isolation_upper=isolation_upper,
            mz_array=mz_array, intensity_array=intensity_array,
        )

    def _load_from_mzml(
        self,
        mzml_file_path: None | str = None
    ):
        """从 mzML 文件加载数据。

        改造说明（spec 2026-06-01-mzml-centroiding-on-load §5.1）：
        第一遍只统计 total_spectra（不访问 peaks 数组，pyteomics 按需懒
        解码）。第二遍 centroid 后通过 chunk + concat 构建
        `_mz_values` / `_intensity_values`；其余按谱图数预分配的数组保持
        现状。
        """
        logging.info(f"Loading DIA data from {mzml_file_path} ...")

        # 第一遍：统计谱图数 + max(scan_id)。不读 peaks 数组（pyteomics
        # 按需懒解码 m/z array 和 intensity array, 只访问 spectrum['id']
        # 几乎不引入额外开销）。max_scan_id 用于正确 size _scan_id_to_index
        # —— 在 pParse / ProteoWizard 过滤后 scan_id 可远大于 total_spectra。
        total_spectra = 0
        max_scan_id = -1
        with mzml.read(mzml_file_path) as reader:
            for spectrum in reader:
                total_spectra += 1
                scan_id = self._extract_scan_number(spectrum['id'])
                if scan_id > max_scan_id:
                    max_scan_id = scan_id

        logging.info(
            f"{mzml_file_path} Total spectra: {total_spectra}, "
            f"max scan_id: {max_scan_id}")

        # 按谱图数 + max_scan_id 预分配定长数组（不再预分配 peak 数组）
        self._preallocate_arrays(total_spectra=total_spectra,
                                 max_scan_id=max_scan_id)

        # 内存权衡（spec §5.1）：chunk + concat 模式下，concat 时同时持
        # 有 chunk list 与新数组，峰值内存约为最终 _mz_values 的 2 倍。
        # 当 _centroid_enabled=True（生产默认值），chunks 已经 centroid
        # 到 5-10% 体积，影响可忽略。当 _centroid_enabled=False（仅调试），
        # profile 数据全量保留，峰值可达数 GB——比旧的单次预分配翻倍。
        # 第二遍：填充。peak 数组通过 chunk list 收集后 concat。
        mz_chunks: list[np.ndarray] = []
        int_chunks: list[np.ndarray] = []
        current_spectrum_idx = 0
        current_peak_idx = 0

        with mzml.read(mzml_file_path) as reader:
            for spectrum in reader:
                mz_chunk, int_chunk = self._process_single_spectrum(
                    spectrum, current_spectrum_idx, current_peak_idx)
                mz_chunks.append(mz_chunk)
                int_chunks.append(int_chunk)

                current_peak_idx += len(mz_chunk)
                current_spectrum_idx += 1

        # Concat peak arrays (一次性, 然后立即释放 chunk list 节省内存)
        self._finalize_arrays(mz_chunks, int_chunks)

    def _load_from_pfb(self, pfb_file_path: str) -> None:
        """从 PFB（pFind/pXtract 二进制）文件加载数据，产出与
        _load_from_mzml 等价的 DIAData。PFB 已是 peak-picked，跳过质心化。"""
        from spectrum import pfb_reader

        logging.info(f"Loading DIA data from {pfb_file_path} (PFB) ...")

        # Pass 1: total_spectra + max scan number（跳过峰，不解码）
        with open(pfb_file_path, "rb") as fh:
            _addr_list_addr, scan_num = pfb_reader.read_header(fh)
            max_scan_id = -1
            for scan in pfb_reader.iter_scan_ids(fh, scan_num):
                if scan > max_scan_id:
                    max_scan_id = scan

        logging.info(
            f"{pfb_file_path} Total spectra: {scan_num}, "
            f"max scan_id: {max_scan_id}")

        self._preallocate_arrays(total_spectra=scan_num,
                                 max_scan_id=max_scan_id)

        # Pass 2: 填充
        mz_chunks: list[np.ndarray] = []
        int_chunks: list[np.ndarray] = []
        current_spectrum_idx = 0
        current_peak_idx = 0

        with open(pfb_file_path, "rb") as fh:
            pfb_reader.read_header(fh)
            for spec in pfb_reader.iter_spectra(fh, scan_num):
                if spec.ms_level == 1:
                    self.has_ms1 = True
                    precursor_scan_id = -1
                    isolation_lower = None
                    isolation_upper = None
                else:
                    precursor_scan_id = spec.precursor_scan
                    if spec.activation_window is None:
                        raise ValueError(
                            f"PFB MS2 scan {spec.scan} missing "
                            f"ActivationWindow; cannot derive DIA isolation "
                            f"window")
                    half = spec.activation_window / 2.0
                    isolation_lower = spec.activation_center - half
                    isolation_upper = spec.activation_center + half

                mz_chunk, int_chunk = self._record_spectrum(
                    current_spectrum_idx, current_peak_idx,
                    scan_id=spec.scan, rt=spec.rt,
                    precursor_scan_id=precursor_scan_id,
                    isolation_lower=isolation_lower,
                    isolation_upper=isolation_upper,
                    mz_array=spec.mz, intensity_array=spec.intensity,
                )
                mz_chunks.append(mz_chunk)
                int_chunks.append(int_chunk)
                current_peak_idx += len(mz_chunk)
                current_spectrum_idx += 1

        self._finalize_arrays(mz_chunks, int_chunks)

    def _finalize_arrays(
        self, mz_chunks: list[np.ndarray], int_chunks: list[np.ndarray]
    ) -> None:
        """加载循环结束后的收尾（格式无关）：concat 峰数组、算 mz 范围、
        ms1/ms2 索引、frame_max_index、DIA 循环左界。"""
        if mz_chunks:
            self._mz_values = np.concatenate(mz_chunks).astype(
                np.float32, copy=False)
            self._intensity_values = np.concatenate(int_chunks).astype(
                np.float32, copy=False)
        else:
            self._mz_values = np.empty(0, dtype=np.float32)
            self._intensity_values = np.empty(0, dtype=np.float32)
        del mz_chunks, int_chunks

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

        self.frame_max_index = len(self.rt_values) - 1

        self.ms2_indexs = np.where(
            self.precursor_scan_ids != -1)[0].astype(np.int32)
        self.ms2_indexs_rt = self.rt_values[self.ms2_indexs].copy()

        if self._precursor_lower_mz is not None:
            self._cycle_left_precursor = deduplicate_with_tolerance(
                self._precursor_lower_mz,
                tolerance=0.1
            )

        if self._n_centroid_empty > 0:
            logging.info(
                "[centroid] %d spectra returned empty (likely <3 peaks "
                "or all-zero intensity)",
                self._n_centroid_empty)

    def check_in_raw(self, precursor_mz) -> bool:
        """ 检查这个 mz 是否在当前 raw 中"""
        if (precursor_mz <= self._max_mz_value + 0.1
                and precursor_mz >= self._min_mz_value - 0.1):
            return True

        # P1-6 (Silent-I3, 2026-06-03 audit): was logging.warn per call
        # (deprecated alias); now debug + counter. Per-worker summary
        # logged at batch end in workflows/flow_utils.py.
        self._n_out_of_window_xic += 1
        logging.debug(
            "out-of-window XIC: max=%s min=%s mz=%s",
            self._max_mz_value, self._min_mz_value, precursor_mz)
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

        # Step 1b: 从 pos 向两侧无界扩展，找最近的、窗口含 precursor_mz 的 MS2 谱图。
        # DIA 每 cycle 轮询 N 个隔离窗口，含 precursor_mz 的窗口每 N 个 MS2 谱图
        # 才出现一次（N 常 20-70）；固定 ±5 在 N>5 时几乎必然漏掉 → 空 XIC。
        # 与下方 Step 2/3 的无界左右收集保持一致。
        n_ms2 = len(self.ms2_indexs_rt)

        def _window_contains(i: int) -> bool:
            gidx = self.ms2_indexs[i]
            lo = self._precursor_lower_mz[gidx]
            up = self._precursor_upper_mz[gidx]
            if np.isnan(lo) or np.isnan(up):
                return False
            return (lo - 0.1) <= precursor_mz <= (up + 0.1)

        center_idx = None
        min_diff = float('inf')
        # 向左（含 pos）找最近的匹配窗口
        i = min(pos, n_ms2 - 1)
        while i >= 0:
            if _window_contains(i):
                center_idx = i
                min_diff = abs(self.ms2_indexs_rt[i] - rt)
                break
            i -= 1
        # 向右（pos 之后）找最近的匹配窗口，与 rt 更近者胜出
        i = pos + 1
        while i < n_ms2:
            if _window_contains(i):
                if abs(self.ms2_indexs_rt[i] - rt) < min_diff:
                    center_idx = i
                break
            i += 1

        if center_idx is None:
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("cycle_idx", "i4")]
            # P1-6 (Silent-I3, 2026-06-03 audit): debug + counter instead
            # of per-call warn.
            self._n_out_of_window_xic += 1
            logging.debug(
                "no MS2 window match: precursor_mz=%s", precursor_mz)
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
