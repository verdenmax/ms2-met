# PFB 谱图格式支持 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 config 的 `raw_path_*` 可填 `.pfb`，经 `DataManager → DIAData` 读出与 mzML 等价的 DIAData。

**Architecture:** 新增纯解析模块 `spectrum/pfb_reader.py`；把 `dia_data.py` 中与格式无关的"写数组"和"收尾"两段抽成共享方法 `_record_spectrum`/`_finalize_arrays`（mzML 与 PFB 共用）；新增 `DIAData._load_from_pfb` 做 2-pass；`get_dia_data_object` 按扩展名分派。

**Tech Stack:** Python 3.14、numpy、`struct`（二进制）、pytest。仅用 `/usr/bin/python3`（已确认 pytest 9.0.3 + pyteomics 可用）。

**关联 spec:** `docs/specs/2026-06-12-pfb-format-support-design.md`

**测试命令前缀（全程使用）:** `/usr/bin/python3 -m pytest`

---

## 文件结构

- 新增 `spectrum/pfb_reader.py` — 纯 PFB 二进制解析（header / property_str / 顺序谱迭代 / scan-id 迭代 / footer）。无 numpy 数组构建、无 DIAData 知识。
- 新增 `tests/pfb_test_helpers.py` — 合成 `.pfb` 写入工具（被两个测试文件共用）。
- 新增 `tests/test_pfb_reader.py` — pfb_reader 单测。
- 新增 `tests/test_dia_data_load_pfb.py` — `_load_from_pfb` + 分派 + 真实文件 opt-in 慢测。
- 修改 `spectrum/dia_data.py` — 抽取 `_record_spectrum`/`_finalize_arrays`，新增 `_load_from_pfb`。
- 修改 `manager/data_manager.py` — 扩展名分派 + `import os`。

> 测试目录是 package（`tests/__init__.py` + `tests/conftest.py` 已存在），跨测试导入用 `from tests.pfb_test_helpers import write_pfb`。

---

## Task 1: 合成 PFB 写入工具 + pfb_reader 骨架与 `read_header`

**Files:**
- Create: `tests/pfb_test_helpers.py`
- Create: `spectrum/pfb_reader.py`
- Test: `tests/test_pfb_reader.py`

- [ ] **Step 1: 写合成 PFB 工具 `tests/pfb_test_helpers.py`**

```python
"""Helpers for building synthetic .pfb files in tests."""
import struct
import numpy as np

_HEADER_SIZE = 24


def make_property_str(spec: dict) -> str:
    """Build a tab-separated property_str from a spec dict (MS1 or MS2)."""
    ms_level = spec["ms_level"]
    parts = [str(spec["scan"]), str(ms_level), str(spec["rt"]),
             spec["instrument_type"]]
    if ms_level == 2:
        parts += [
            str(spec["charge"]),
            str(spec["mh_plus"]),
            str(spec["ion_injection_time"]),
            str(spec["activation_center"]),
            spec["activation_type"],
            str(spec["precursor_scan"]),
            str(spec["activation_window"]),
            str(spec["nce"]),
            str(spec["monoisotopic_mz"]),
        ]
    return "\t".join(parts)


def write_pfb(path, spectra, empties=(0, 0, 0)):
    """Write a synthetic .pfb file. Returns the footer addr_list (offsets).

    Each spec dict: scan, ms_level, rt, instrument_type, mz(list), intensity(list).
    MS2 adds: charge, mh_plus, ion_injection_time, activation_center,
    activation_type, precursor_scan, activation_window, nce, monoisotopic_mz.
    """
    addr_list = []
    body = bytearray()
    for spec in spectra:
        addr_list.append(_HEADER_SIZE + len(body))
        pstr = make_property_str(spec).encode("utf-8")
        body += struct.pack("<i", len(pstr))
        body += pstr
        mz = np.asarray(spec["mz"], dtype="<f8")
        inten = np.asarray(spec["intensity"], dtype="<f8")
        assert len(mz) == len(inten)
        body += struct.pack("<i", len(mz))
        body += mz.tobytes()
        body += inten.tobytes()
    addr_list_addr = _HEADER_SIZE + len(body)
    with open(path, "wb") as f:
        f.write(struct.pack("<iiiqi", empties[0], empties[1], empties[2],
                            addr_list_addr, len(spectra)))
        f.write(body)
        if spectra:
            f.write(struct.pack(f"<{len(spectra)}q", *addr_list))
    return addr_list
```

- [ ] **Step 2: 写失败测试 `tests/test_pfb_reader.py`（read_header + 24 字节头）**

```python
"""Tests for spectrum.pfb_reader."""
import struct

import numpy as np
import pytest

from spectrum import pfb_reader
from tests.pfb_test_helpers import write_pfb

_MS1 = {"scan": 1, "ms_level": 1, "rt": 1.5, "instrument_type": "FTMS",
        "mz": [350.0, 351.0], "intensity": [10.0, 20.0]}
_MS2 = {"scan": 2, "ms_level": 2, "rt": 2.0, "instrument_type": "FTMS",
        "charge": 2, "mh_plus": 1000.5, "ion_injection_time": 63.0,
        "activation_center": 501.0, "activation_type": "HCD",
        "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
        "monoisotopic_mz": 501.0, "mz": [100.0, 101.0, 102.0],
        "intensity": [5.0, 6.0, 7.0]}


def test_read_header_returns_addr_and_scan_num(tmp_path):
    p = tmp_path / "x.pfb"
    addr_list = write_pfb(str(p), [_MS1, _MS2])
    with open(p, "rb") as fh:
        addr_list_addr, scan_num = pfb_reader.read_header(fh)
        assert scan_num == 2
        assert addr_list_addr > addr_list[-1]  # footer starts after last spectrum
        # header is exactly 24 bytes -> first spectrum at offset 24
        assert pfb_reader.HEADER_SIZE == 24
        assert fh.tell() == 24
        assert addr_list[0] == 24
```

- [ ] **Step 3: 运行，确认失败**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'spectrum.pfb_reader'`）

- [ ] **Step 4: 写 `spectrum/pfb_reader.py` 骨架 + read_header**

```python
"""PFB (pFind/pXtract binary spectrum) format reader.

Pure parsing: reads the binary structure into typed per-spectrum records.
No numpy-array-building / DIAData knowledge lives here.

Format (little-endian), verified against real samples:
  Header (24 bytes): 3xint32 (reserved) + int64 addr_list_addr + int32 scan_num
  Loop body x scan_num:
    int32 property_str_len
    char[property_str_len]  property_str (UTF-8, '\\t'-separated, may end \\x00)
    int32 peak_num
    float64[peak_num]  mz
    float64[peak_num]  intensity
  Footer: int64[scan_num]  addr_list (per-spectrum file offsets)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterator

import numpy as np

_HEADER_STRUCT = struct.Struct("<iiiqi")
HEADER_SIZE = _HEADER_STRUCT.size  # 24

_MS1_FIELD_COUNT = 4
_MS2_FIELD_COUNT = 13


@dataclass
class PFBSpectrum:
    scan: int
    ms_level: int
    rt: float
    instrument_type: str
    mz: np.ndarray
    intensity: np.ndarray
    charge: int | None = None
    mh_plus: float | None = None
    ion_injection_time: float | None = None
    activation_center: float | None = None
    activation_type: str | None = None
    precursor_scan: int | None = None
    activation_window: float | None = None
    nce: float | None = None
    monoisotopic_mz: float | None = None


def read_header(fh: BinaryIO) -> tuple[int, int]:
    """Read the 24-byte header. Returns (addr_list_addr, scan_num).

    Leaves the file positioned at the first spectrum (offset 24).
    """
    raw = fh.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"PFB header truncated: expected {HEADER_SIZE} bytes, "
            f"got {len(raw)}")
    _e1, _e2, _e3, addr_list_addr, scan_num = _HEADER_STRUCT.unpack(raw)
    return addr_list_addr, scan_num
```

- [ ] **Step 5: 运行，确认通过**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py -q`
Expected: PASS（1 passed）

- [ ] **Step 6: 提交**

```bash
git add spectrum/pfb_reader.py tests/pfb_test_helpers.py tests/test_pfb_reader.py
git commit -m "feat(pfb): synthetic .pfb writer helper + read_header"
```

---

## Task 2: `parse_property_str`（按 MsType 分派 MS1/MS2 布局）

**Files:**
- Modify: `spectrum/pfb_reader.py`
- Test: `tests/test_pfb_reader.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_parse_property_str_ms1():
    out = pfb_reader.parse_property_str("1\t1\t0.197\tFTMS")
    assert out == {"scan": 1, "ms_level": 1, "rt": 0.197,
                   "instrument_type": "FTMS"}


def test_parse_property_str_ms2():
    s = "2\t2\t0.4538569\tFTMS\t2\t1000.993\t63\t501\tHCD\t1\t2\t27.00\t501"
    out = pfb_reader.parse_property_str(s)
    assert out["scan"] == 2 and out["ms_level"] == 2
    assert out["instrument_type"] == "FTMS"
    assert out["charge"] == 2
    assert out["activation_center"] == 501.0
    assert out["precursor_scan"] == 1
    assert out["activation_window"] == 2.0
    assert out["nce"] == 27.0
    assert out["monoisotopic_mz"] == 501.0


def test_parse_property_str_ms2_wrong_field_count_raises():
    # MS2 with only 11 fields (missing pXtract-3 fields) -> clear error
    s = "2\t2\t0.45\tFTMS\t2\t1000.9\t63\t501\tHCD\t1\t2"
    with pytest.raises(ValueError, match="MS2"):
        pfb_reader.parse_property_str(s)
```

- [ ] **Step 2: 运行，确认失败**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py -q -k parse_property_str`
Expected: FAIL（`AttributeError: module 'spectrum.pfb_reader' has no attribute 'parse_property_str'`）

- [ ] **Step 3: 在 `spectrum/pfb_reader.py` 追加 `parse_property_str`**

```python
def parse_property_str(s: str) -> dict:
    """Parse a tab-separated property string into typed fields.

    Layout decided by token[1] (MsType): MS1 -> 4 tokens, MS2 -> 13 tokens.
    """
    toks = s.split("\t")
    if len(toks) < _MS1_FIELD_COUNT:
        raise ValueError(f"PFB property_str has too few fields: {toks!r}")
    ms_level = int(toks[1])
    base = {
        "scan": int(toks[0]),
        "ms_level": ms_level,
        "rt": float(toks[2]),
        "instrument_type": toks[3],
    }
    if ms_level == 1:
        if len(toks) != _MS1_FIELD_COUNT:
            raise ValueError(
                f"MS1 property_str expects {_MS1_FIELD_COUNT} fields, "
                f"got {len(toks)}: {toks!r}")
        return base
    if ms_level == 2:
        if len(toks) != _MS2_FIELD_COUNT:
            raise ValueError(
                f"MS2 property_str expects {_MS2_FIELD_COUNT} fields, "
                f"got {len(toks)}: {toks!r}")
        base.update({
            "charge": int(toks[4]),
            "mh_plus": float(toks[5]),
            "ion_injection_time": float(toks[6]),
            "activation_center": float(toks[7]),
            "activation_type": toks[8],
            "precursor_scan": int(toks[9]),
            "activation_window": float(toks[10]),
            "nce": float(toks[11]),
            "monoisotopic_mz": float(toks[12]),
        })
        return base
    raise ValueError(f"Unknown MsType={ms_level} in property_str: {toks!r}")
```

- [ ] **Step 4: 运行，确认通过**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add spectrum/pfb_reader.py tests/test_pfb_reader.py
git commit -m "feat(pfb): parse_property_str with MS1/MS2 layout dispatch"
```

---

## Task 3: `_read_exact` + `iter_spectra`（顺序读 loop body）

**Files:**
- Modify: `spectrum/pfb_reader.py`
- Test: `tests/test_pfb_reader.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_iter_spectra_yields_ms1_and_ms2(tmp_path):
    p = tmp_path / "x.pfb"
    write_pfb(str(p), [_MS1, _MS2])
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        specs = list(pfb_reader.iter_spectra(fh, scan_num))
    assert len(specs) == 2
    s1, s2 = specs
    assert s1.ms_level == 1 and s1.scan == 1 and s1.rt == 1.5
    np.testing.assert_allclose(s1.mz, [350.0, 351.0])
    np.testing.assert_allclose(s1.intensity, [10.0, 20.0])
    assert s1.charge is None
    assert s2.ms_level == 2 and s2.precursor_scan == 1
    assert s2.activation_center == 501.0 and s2.activation_window == 2.0
    np.testing.assert_allclose(s2.mz, [100.0, 101.0, 102.0])
    np.testing.assert_allclose(s2.intensity, [5.0, 6.0, 7.0])
    assert s2.mz.dtype == np.float64


def test_iter_spectra_empty_file(tmp_path):
    p = tmp_path / "empty.pfb"
    write_pfb(str(p), [])
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        assert scan_num == 0
        assert list(pfb_reader.iter_spectra(fh, scan_num)) == []


def test_iter_spectra_truncated_raises(tmp_path):
    p = tmp_path / "trunc.pfb"
    write_pfb(str(p), [_MS1, _MS2])
    # Truncate the file mid-body
    full = p.read_bytes()
    p.write_bytes(full[:30])
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        with pytest.raises(ValueError, match="truncated"):
            list(pfb_reader.iter_spectra(fh, scan_num))
```

- [ ] **Step 2: 运行，确认失败**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py -q -k iter_spectra`
Expected: FAIL（`AttributeError: ... has no attribute 'iter_spectra'`）

- [ ] **Step 3: 在 `spectrum/pfb_reader.py` 追加 `_read_exact` + `iter_spectra`**

```python
def _read_exact(fh: BinaryIO, n: int, spec_idx: int, what: str) -> bytes:
    raw = fh.read(n)
    if len(raw) < n:
        raise ValueError(
            f"PFB truncated reading spectrum {spec_idx} {what}: "
            f"want {n} bytes, got {len(raw)} at offset {fh.tell()}")
    return raw


def iter_spectra(fh: BinaryIO, scan_num: int) -> Iterator[PFBSpectrum]:
    """Sequentially read `scan_num` spectra from the loop body.

    `fh` must be positioned at the first spectrum (call read_header first).
    """
    for i in range(scan_num):
        (slen,) = struct.unpack("<i", _read_exact(fh, 4, i, "property_str_len"))
        prop = _read_exact(fh, slen, i, "property_str").decode(
            "utf-8").rstrip("\x00")
        fields = parse_property_str(prop)
        (pnum,) = struct.unpack("<i", _read_exact(fh, 4, i, "peak_num"))
        if pnum > 0:
            mz = np.frombuffer(
                _read_exact(fh, pnum * 8, i, "mz"), dtype="<f8").astype(
                np.float64)
            intensity = np.frombuffer(
                _read_exact(fh, pnum * 8, i, "intensity"), dtype="<f8").astype(
                np.float64)
        else:
            mz = np.empty(0, dtype=np.float64)
            intensity = np.empty(0, dtype=np.float64)
        yield PFBSpectrum(mz=mz, intensity=intensity, **fields)
```

- [ ] **Step 4: 运行，确认通过**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py -q`
Expected: PASS（7 passed）

- [ ] **Step 5: 提交**

```bash
git add spectrum/pfb_reader.py tests/test_pfb_reader.py
git commit -m "feat(pfb): iter_spectra sequential loop-body reader (intensity=double)"
```

---

## Task 4: `iter_scan_ids`（pass-1，跳过峰）+ `read_footer`（自检用）

**Files:**
- Modify: `spectrum/pfb_reader.py`
- Test: `tests/test_pfb_reader.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_iter_scan_ids_skips_peaks(tmp_path):
    p = tmp_path / "x.pfb"
    write_pfb(str(p), [_MS1, _MS2])
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        scans = list(pfb_reader.iter_scan_ids(fh, scan_num))
    assert scans == [1, 2]


def test_read_footer_matches_offsets(tmp_path):
    p = tmp_path / "x.pfb"
    addr_list = write_pfb(str(p), [_MS1, _MS2])
    with open(p, "rb") as fh:
        addr_list_addr, scan_num = pfb_reader.read_header(fh)
        footer = pfb_reader.read_footer(fh, addr_list_addr, scan_num)
    assert footer == addr_list
    assert footer[0] == 24
```

- [ ] **Step 2: 运行，确认失败**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py -q -k "iter_scan_ids or read_footer"`
Expected: FAIL（`AttributeError: ... has no attribute 'iter_scan_ids'`）

- [ ] **Step 3: 在 `spectrum/pfb_reader.py` 追加 `iter_scan_ids` + `read_footer`**

```python
def iter_scan_ids(fh: BinaryIO, scan_num: int) -> Iterator[int]:
    """Pass-1: yield each spectrum's scan number, seeking past peak arrays.

    `fh` must be positioned at the first spectrum (call read_header first).
    Does NOT decode peak arrays (cheap two-pass like the mzML loader).
    """
    for i in range(scan_num):
        (slen,) = struct.unpack("<i", _read_exact(fh, 4, i, "property_str_len"))
        prop = _read_exact(fh, slen, i, "property_str").decode(
            "utf-8").rstrip("\x00")
        scan = int(prop.split("\t", 1)[0])
        (pnum,) = struct.unpack("<i", _read_exact(fh, 4, i, "peak_num"))
        fh.seek(pnum * 16, 1)  # skip mz(8) + intensity(8) per peak
        yield scan


def read_footer(fh: BinaryIO, addr_list_addr: int, scan_num: int) -> list[int]:
    """Read the footer addr_list (per-spectrum file offsets). For validation."""
    fh.seek(addr_list_addr)
    raw = fh.read(scan_num * 8)
    if len(raw) < scan_num * 8:
        raise ValueError(
            f"PFB footer truncated: want {scan_num * 8} bytes, got {len(raw)}")
    return list(struct.unpack(f"<{scan_num}q", raw))
```

- [ ] **Step 4: 运行，确认通过**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py -q`
Expected: PASS（9 passed）

- [ ] **Step 5: 提交**

```bash
git add spectrum/pfb_reader.py tests/test_pfb_reader.py
git commit -m "feat(pfb): iter_scan_ids (pass-1) + read_footer for validation"
```

---

## Task 5: 抽取共享 `_record_spectrum` + `_finalize_arrays`（mzML 重构，行为不变）

**Files:**
- Modify: `spectrum/dia_data.py`（`_process_single_spectrum` 尾部 538–561；`_load_from_mzml` 收尾 620–667）
- Test: 现有 `tests/test_dia_data_load_mzml.py`、`tests/test_dia_data_window.py`（守护，不新增）

- [ ] **Step 1: 先跑现有测试建立基线**

Run: `/usr/bin/python3 -m pytest tests/test_dia_data_load_mzml.py tests/test_dia_data_window.py -q`
Expected: PASS（28 passed）

- [ ] **Step 2: 在 `DIAData` 中新增 `_record_spectrum` 方法**

在 `spectrum/dia_data.py` 的 `_process_single_spectrum` 方法**之前**插入：

```python
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
```

- [ ] **Step 3: 改写 `_process_single_spectrum` 尾部调用共享方法**

把 `_process_single_spectrum` 末尾（现 538–561 行，从 `peak_stop_idx = current_peak_index + len(mz_array)` 到 `return mz_array, intensity_array`）替换为：

```python
        return self._record_spectrum(
            spectrum_idx, current_peak_index,
            scan_id=scan_id, rt=rt,
            precursor_scan_id=precursor_scan_id,
            isolation_lower=isolation_lower,
            isolation_upper=isolation_upper,
            mz_array=mz_array, intensity_array=intensity_array,
        )
```

> 注意：保留该方法前面"`del spectrum`"和"`if ms_level == 1: self.has_ms1 = True`"等逻辑不动；只替换从 `peak_stop_idx = ...` 开始到 `return` 的那段。

- [ ] **Step 4: 在 `DIAData` 中新增 `_finalize_arrays` 方法**

在 `_load_from_mzml` 方法**之后**插入（方法体即现 620–667 行的收尾逻辑）：

```python
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
```

- [ ] **Step 5: 改写 `_load_from_mzml` 收尾调用共享方法**

把 `_load_from_mzml` 中从 `if mz_chunks:`（现 620 行）到方法结束的整段收尾，替换为一行：

```python
        self._finalize_arrays(mz_chunks, int_chunks)
```

- [ ] **Step 6: 运行现有测试，确认行为不变**

Run: `/usr/bin/python3 -m pytest tests/test_dia_data_load_mzml.py tests/test_dia_data_window.py tests/test_dia_cache.py -q`
Expected: PASS（全绿，0 失败）

- [ ] **Step 7: 提交**

```bash
git add spectrum/dia_data.py
git commit -m "refactor(dia_data): extract _record_spectrum + _finalize_arrays (format-agnostic)"
```

---

## Task 6: `DIAData._load_from_pfb`（2-pass，复用共享方法）

**Files:**
- Modify: `spectrum/dia_data.py`（新增 `_load_from_pfb`）
- Test: `tests/test_dia_data_load_pfb.py`

- [ ] **Step 1: 写失败测试 `tests/test_dia_data_load_pfb.py`**

```python
"""Tests for DIAData._load_from_pfb."""
import numpy as np
import pytest

from spectrum.dia_data import DIAData
from tests.pfb_test_helpers import write_pfb

_MS1 = {"scan": 1, "ms_level": 1, "rt": 1.0, "instrument_type": "FTMS",
        "mz": [350.0, 351.0], "intensity": [10.0, 20.0]}
_MS2A = {"scan": 2, "ms_level": 2, "rt": 1.1, "instrument_type": "FTMS",
         "charge": 2, "mh_plus": 1000.5, "ion_injection_time": 63.0,
         "activation_center": 501.0, "activation_type": "HCD",
         "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
         "monoisotopic_mz": 501.0, "mz": [100.0, 101.0, 102.0],
         "intensity": [5.0, 6.0, 7.0]}
_MS2B = {"scan": 3, "ms_level": 2, "rt": 1.2, "instrument_type": "FTMS",
         "charge": 3, "mh_plus": 1500.0, "ion_injection_time": 50.0,
         "activation_center": 503.0, "activation_type": "HCD",
         "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
         "monoisotopic_mz": 503.0, "mz": [200.0], "intensity": [9.0]}


def test_load_from_pfb_builds_equivalent_arrays(tmp_path):
    p = tmp_path / "x.pfb"
    write_pfb(str(p), [_MS1, _MS2A, _MS2B])
    d = DIAData()
    d._load_from_pfb(str(p))

    # peak arrays concatenated in order, stored float32
    np.testing.assert_allclose(
        d._mz_values, [350.0, 351.0, 100.0, 101.0, 102.0, 200.0])
    np.testing.assert_allclose(
        d._intensity_values, [10.0, 20.0, 5.0, 6.0, 7.0, 9.0])
    assert d._mz_values.dtype == np.float32

    # per-spectrum slices
    np.testing.assert_array_equal(d._peak_start_idx_list, [0, 2, 5])
    np.testing.assert_array_equal(d._peak_stop_idx_list, [2, 5, 6])

    # ms1/ms2 split via precursor_scan_ids
    np.testing.assert_array_equal(d.precursor_scan_ids, [-1, 1, 1])
    np.testing.assert_array_equal(d.ms1_indexs, [0])
    np.testing.assert_array_equal(d.ms2_indexs, [1, 2])

    # RT preserved (seconds, no conversion)
    np.testing.assert_allclose(d.rt_values, [1.0, 1.1, 1.2], rtol=1e-6)

    # DIA window = activation_center +/- activation_window/2
    assert np.isnan(d._precursor_lower_mz[0])  # MS1
    np.testing.assert_allclose(d._precursor_lower_mz[1:], [500.0, 502.0])
    np.testing.assert_allclose(d._precursor_upper_mz[1:], [502.0, 504.0])
    assert float(d._min_mz_value) == pytest.approx(500.0)
    assert float(d._max_mz_value) == pytest.approx(504.0)

    # scan_id -> index map (scan numbers 1,2,3 -> idx 0,1,2)
    assert d._scan_id_to_index[1] == 0
    assert d._scan_id_to_index[2] == 1
    assert d._scan_id_to_index[3] == 2
    assert d.has_ms1 is True


def test_load_from_pfb_empty_file(tmp_path):
    p = tmp_path / "empty.pfb"
    write_pfb(str(p), [])
    d = DIAData()
    d._load_from_pfb(str(p))
    assert len(d._mz_values) == 0
    assert len(d.ms1_indexs) == 0
    assert len(d.ms2_indexs) == 0
```

- [ ] **Step 2: 运行，确认失败**

Run: `/usr/bin/python3 -m pytest tests/test_dia_data_load_pfb.py -q`
Expected: FAIL（`AttributeError: 'DIAData' object has no attribute '_load_from_pfb'`）

- [ ] **Step 3: 在 `spectrum/dia_data.py` 新增 `_load_from_pfb`**

在 `_load_from_mzml` 方法之后（`_finalize_arrays` 附近）插入：

```python
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
```

- [ ] **Step 4: 运行，确认通过**

Run: `/usr/bin/python3 -m pytest tests/test_dia_data_load_pfb.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add spectrum/dia_data.py tests/test_dia_data_load_pfb.py
git commit -m "feat(pfb): DIAData._load_from_pfb (2-pass, reuses shared helpers)"
```

---

## Task 7: `get_dia_data_object` 按扩展名分派

**Files:**
- Modify: `manager/data_manager.py`（加 `import os` + 分派）
- Test: `tests/test_dia_data_load_pfb.py`

- [ ] **Step 1: 追加失败测试（验证分派路由，不读真实文件）**

```python
def test_get_dia_data_object_dispatches_by_extension(monkeypatch):
    from manager.data_manager import DataManager

    called = {}

    def fake_pfb(self, path):
        called["pfb"] = path

    def fake_mzml(self, path):
        called["mzml"] = path

    monkeypatch.setattr(DIAData, "_load_from_pfb", fake_pfb)
    monkeypatch.setattr(DIAData, "_load_from_mzml", fake_mzml)

    dm = DataManager(config=None, path=None)
    dm.get_dia_data_object("/tmp/sample.pfb")
    assert called == {"pfb": "/tmp/sample.pfb"}

    called.clear()
    dm.get_dia_data_object("/tmp/sample.mzML")
    assert called == {"mzml": "/tmp/sample.mzML"}
```

- [ ] **Step 2: 运行，确认失败**

Run: `/usr/bin/python3 -m pytest tests/test_dia_data_load_pfb.py -q -k dispatch`
Expected: FAIL（当前 `get_dia_data_object` 无条件调用 `_load_from_mzml`，`.pfb` 分支断言失败：`called == {"mzml": "/tmp/sample.pfb"}`）

- [ ] **Step 3: 修改 `manager/data_manager.py`**

在文件顶部 import 区加入（与现有 `import configparser` 同段）：

```python
import os
```

把 `get_dia_data_object` 中这一行：

```python
        dia_data._load_from_mzml(tot_raw_path)
```

替换为：

```python
        ext = os.path.splitext(tot_raw_path or "")[1].lower()
        if ext == ".pfb":
            dia_data._load_from_pfb(tot_raw_path)
        else:
            dia_data._load_from_mzml(tot_raw_path)
```

- [ ] **Step 4: 运行，确认通过**

Run: `/usr/bin/python3 -m pytest tests/test_dia_data_load_pfb.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add manager/data_manager.py tests/test_dia_data_load_pfb.py
git commit -m "feat(pfb): dispatch .pfb vs mzML by extension in get_dia_data_object"
```

---

## Task 8: 真实文件 opt-in 慢测 + 全量回归

**Files:**
- Modify: `tests/test_dia_data_load_pfb.py`（加真实文件 opt-in 测试）

- [ ] **Step 1: 追加真实文件 opt-in 测试**

```python
import os

_REAL_PFB = os.path.expanduser(
    "~/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/2th/"
    "20190830_HF_ZHW_hela_SILAC_DDIA_500_550_2Da_Rep1.pfb")


@pytest.mark.skipif(not os.path.exists(_REAL_PFB),
                    reason="real .pfb sample not available")
def test_real_pfb_header_and_first_spectra():
    from spectrum import pfb_reader
    with open(_REAL_PFB, "rb") as fh:
        addr_list_addr, scan_num = pfb_reader.read_header(fh)
        assert scan_num == 80096
        specs = []
        for s in pfb_reader.iter_spectra(fh, scan_num):
            specs.append(s)
            if len(specs) >= 2:
                break
        footer = pfb_reader.read_footer(fh, addr_list_addr, scan_num)
    s1, s2 = specs
    # first spectrum is MS1, RT ~ 0.197 sec
    assert s1.ms_level == 1
    assert s1.rt == pytest.approx(0.1972939, rel=1e-4)
    # second is MS2 in a 2Da window centred at 501 -> [500, 502]
    assert s2.ms_level == 2
    assert s2.activation_center == pytest.approx(501.0)
    assert s2.activation_window == pytest.approx(2.0)
    # footer integrity: first offset == header size
    assert footer[0] == pfb_reader.HEADER_SIZE
```

- [ ] **Step 2: 运行（有样例则跑，无则 skip）**

Run: `/usr/bin/python3 -m pytest tests/test_dia_data_load_pfb.py -q`
Expected: PASS（前 3 个通过；真实文件测试在本机有样例 → 也 PASS；CI 无样例 → skipped）

- [ ] **Step 3: 全量 PFB + 受影响测试回归**

Run: `/usr/bin/python3 -m pytest tests/test_pfb_reader.py tests/test_dia_data_load_pfb.py tests/test_dia_data_load_mzml.py tests/test_dia_data_window.py tests/test_dia_cache.py -q`
Expected: PASS（无新增失败；mzML/window/cache 全绿）

- [ ] **Step 4: 提交**

```bash
git add tests/test_dia_data_load_pfb.py
git commit -m "test(pfb): opt-in real-file smoke test (scan_num/RT/window/footer)"
```

- [ ] **Step 5: 推送两个远端**

```bash
git push origin feature_extraction
git push gitlab feature_extraction
```

---

## 验证清单（实现完成后）

- [ ] `tests/test_pfb_reader.py`（9）+ `tests/test_dia_data_load_pfb.py`（4，含 1 opt-in）全绿。
- [ ] 现有 `tests/test_dia_data_load_mzml.py`(19) + `tests/test_dia_data_window.py`(9) + `tests/test_dia_cache.py` 全绿（重构未回归）。
- [ ] 真实样例（本机存在）：`test_real_pfb_header_and_first_spectra` 通过（scan_num=80096、首 MS1 RT≈0.197、首 MS2 窗 [500,502]、footer[0]==24）。
- [ ] 缓存验证（手动，可选）：把某个 baseline 的 `config.ini` 的 `raw_path_*` 指向 `.pfb`，跑一次特征提取确认 `save_to_file(source_path=.pfb)` 的 npz 缓存键用 `.pfb` 路径（spec §5 验证点）。
