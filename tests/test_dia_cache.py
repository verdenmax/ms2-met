"""npz 缓存健壮性：源文件身份校验 + 损坏缓存重建 + 原子写。"""
import os
import numpy as np
import pytest
from spectrum.dia_data import DIAData
from workflows.flow_utils import data_to_npz


def _minimal_dia():
    d = DIAData.__new__(DIAData)
    d.has_mobility = False
    d.has_ms1 = True
    d._max_mz_value = 1000.0
    d._min_mz_value = 400.0
    d._zeroth_frame = 0
    d._scan_max_index = 1
    d.frame_max_index = 2
    d.ms1_indexs = np.array([0], dtype=np.int32)
    d.ms1_indexs_rt = np.array([1.0], dtype=np.float32)
    d.ms2_indexs = np.array([1, 2], dtype=np.int32)
    d.ms2_indexs_rt = np.array([1.1, 1.2], dtype=np.float32)
    d.precursor_scan_ids = np.array([-1, 100, 100], dtype=np.int64)
    d._mz_values = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    d.rt_values = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    d._intensity_values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    d.mobility_values = np.array([1e-6, 0.0], dtype=np.float32)
    d._cycle_left_precursor = np.array([400.0, 500.0], dtype=np.float32)
    d._scan_id_to_index = np.array([0, 1, 2], dtype=np.int64)
    d._peak_start_idx_list = np.array([0, 1, 2], dtype=np.int64)
    d._peak_stop_idx_list = np.array([1, 2, 3], dtype=np.int64)
    d._precursor_lower_mz = np.array([np.nan, 400.0, 500.0], dtype=np.float32)
    d._precursor_upper_mz = np.array([np.nan, 500.0, 600.0], dtype=np.float32)
    d._quad_max_mz_value = None
    d._quad_min_mz_value = None
    d._centroid_enabled = True
    d._centroid_rel_threshold = 1e-3
    return d


def _validate(npz, src=None):
    DIAData.validate_cache_params(
        str(npz), expected_centroid_enabled=True,
        expected_centroid_rel_threshold=1e-3, expected_source_path=src)


def test_cache_invalidated_when_source_changes(tmp_path):
    src = tmp_path / "raw.mzML"
    src.write_bytes(b"abc")
    npz = tmp_path / "raw.dia.npz"
    _minimal_dia().save_to_file(str(npz), source_path=str(src))
    _validate(npz, str(src))                 # 匹配 → 不抛
    src.write_bytes(b"abcdefghij")           # 源文件变化（size 变）
    with pytest.raises(ValueError, match="源文件"):
        _validate(npz, str(src))


def test_cache_backcompat_no_source_fields(tmp_path):
    """旧缓存无源身份字段时，源校验应跳过（保持命中），不报错。"""
    npz = tmp_path / "raw.dia.npz"
    _minimal_dia().save_to_file(str(npz))    # 不带 source_path → 无身份字段
    _validate(npz, str(tmp_path / "raw.mzML"))   # 应不抛


def test_save_atomic_no_tmp_leftover_and_loadable(tmp_path):
    npz = tmp_path / "raw.dia.npz"
    _minimal_dia().save_to_file(str(npz))
    assert npz.exists()
    assert not any(p.name.endswith(".tmp.npz") or ".tmp" in p.name
                   for p in tmp_path.iterdir())
    DIAData.load_from_file(str(npz), use_mmap=False)   # 可加载


class _FakeMgr:
    def get_centroid_params(self):
        return (True, 1e-3)

    def get_dia_data_object(self, filepath):
        return _minimal_dia()


def test_data_to_npz_rebuilds_on_corrupt_cache(tmp_path):
    """损坏的 npz（非 ValueError，如 BadZipFile）应触发重建而非崩溃。"""
    src = tmp_path / "raw.mzML"
    src.write_bytes(b"abc")
    corrupt = tmp_path / "raw.dia.npz"
    corrupt.write_bytes(b"PK\x03\x04" + b"\x00" * 40)   # zip 头但截断 → BadZipFile
    name, shared = data_to_npz(_FakeMgr(), str(src), str(tmp_path))
    assert name == "raw"
    # 重建后应可正常加载
    DIAData.load_from_file(shared, use_mmap=False)
