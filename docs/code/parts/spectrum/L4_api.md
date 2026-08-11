# spectrum — API 参考

逐文件列出主要 class / 函数。`speclib` 子包另见其文档。

## spectrum/dia_data.py

### 模块常量
- `DEFAULT_VALUE_NO_MOBILITY = 1e-6`
- `DEFAULT_CENTROID_ENABLED = True`、`DEFAULT_CENTROID_REL_THRESHOLD = 1e-3`（centroid 配置单一来源）

### `deduplicate_with_tolerance(arr, tolerance=0.1) -> np.ndarray | None`
float32 数组排序后容差去重；`None`/空 → `None`。用于构造 DIA 窗口左边界集合。

### `class DIAData`
一个 raw 的全部 DIA 谱图与 DIA 窗口信息，峰数据以全局扁平数组 + 谱图级 start/stop 索引存储。

主要属性：`_mz_values`/`_intensity_values`（全局峰）、`rt_values`（分钟）、`precursor_scan_ids`（MS1=-1）、`_peak_start_idx_list`/`_peak_stop_idx_list`、`_precursor_lower_mz`/`_precursor_upper_mz`、`_scan_id_to_index`、`ms1_indexs`/`ms2_indexs`(+`_rt`)、`_cycle_left_precursor`、`_centroid_enabled`/`_centroid_rel_threshold`。

- `save_to_file(filepath: str, source_path: str | None = None)` — 存为 `.npz`（`savez_compressed`，`_format_version=3`，过滤 None，原子写 `mkstemp`+`os.replace`）。给定 `source_path` 时把源文件 `mtime`/`size` 写入缓存（`_source_mtime`/`_source_size`），供 `validate_cache_params` 检测源文件是否被替换/重建。
- `classmethod load_from_file(filepath, use_mmap=True, expected_centroid_enabled=None, expected_centroid_rel_threshold=None) -> DIAData` — 从 npz 加载；**Raises** `ValueError`（版本≠3 或 centroid 参数不符）。
- `staticmethod validate_cache_params(filepath, expected_centroid_enabled, expected_centroid_rel_threshold, expected_source_path=None) -> None` — 用 mmap 只读元数据标量做轻量校验：版本、centroid 参数，以及（给定 `expected_source_path` 且缓存含 `_source_size` 且源文件存在时）源文件 size/mtime；不符 **Raises** `ValueError`。源文件校验**被跳过**时（旧缓存无字段 / 源文件不存在）发 `WARNING`，不静默通过。
- `_load_from_mzml(mzml_file_path=None)` — 两遍加载 mzML（统计→填充+质心化+concat）。
- `_load_from_pfb(pfb_file_path)` — 两遍加载 PFB（统计→填充+concat）。**不质心化**（PFB 已 peak-picked）；RT 秒→分钟 `/60`；MS2 隔离窗口 = `activation_center ± activation_window/2`；MS1 `precursor_scan_id=-1`。复用 `_record_spectrum` / `_finalize_arrays`。
- `_record_spectrum(spectrum_idx, current_peak_index, *, scan_id, rt, precursor_scan_id, isolation_lower, isolation_upper, mz_array, intensity_array) -> (mz, intensity)` — 格式无关：把单谱归一化字段写入按谱图定长的数组（isolation 为 None 时存 NaN）。mzML 与 PFB 共用。
- `_finalize_arrays(mz_chunks, int_chunks) -> None` — 格式无关收尾：concat→float32、mz 范围、`ms1_indexs`/`ms2_indexs`(+`_rt`)、`frame_max_index`、`_cycle_left_precursor`。mzML 与 PFB 共用。
- `get_spectrum_by_index(index) -> (mz, intensity)` — 按谱图下标切片；越界 **Raises** `IndexError`。
- `get_spectrum(scan_id) -> (mz, intensity)` — 经 `_scan_id_to_index` 反查。
- `get_ms1_spectrum_by_ms1_index(index) -> (mz, intensity)` — 由 MS2 下标取其前体 MS1。
- `get_spectrum_by_rt(rt, precurso_mz) -> (mz, intensity)` — `searchsorted` 取谱图（假设 RT 递增）。
- `check_in_raw(precursor_mz) -> bool` — m/z 是否在 raw 范围（±0.1）；越界计数 `_n_out_of_window_xic`。
- `check_in_same_ms2(p1, p2) -> bool` — 两前体是否落同一 MS2 窗口。
- `get_window_info(precursor_mz) -> dict` — `{"width","centering","lower","upper"}`，未命中返回默认。
- `find_near_ms1_idx(rt) -> int` — 最近 MS1 在 `ms1_indexs` 中的位置。

- `xic_ms2_peaks_extract(rt, xic_cycle_window, precursor_mz, ions_mass, mass_tol_ppm) -> (np.ndarray, float)`
  - 参数：`rt` 目标保留时间；`xic_cycle_window` 左右各扩展的有效 MS2 谱图数；`precursor_mz` 用于选 DIA 窗口；`ions_mass` 碎片离子中性质量（对 charge 1,2 算理论 m/z）；`mass_tol_ppm` 匹配容差。
  - 返回：结构化数组 `[(rt f8, ppm_error f8, intensity f8, cycle_idx i4)]` + 全窗口 total_intensity。无 MS2 或无窗口匹配返回空数组 + 0.0。

- `xic_peaks_extreact(rt, xic_cycle_window, precursor_mz, mass_tol_ppm) -> np.ndarray`
  - MS1 前体 XIC，返回同上结构化数组（无 total_intensity 第二返回值）。

示例：
```python
dia = DIAData.load_from_file("run.npz", expected_centroid_enabled=True,
                             expected_centroid_rel_threshold=1e-3)
arr, total = dia.xic_ms2_peaks_extract(rt=33.5, xic_cycle_window=5,
                                       precursor_mz=650.3, ions_mass=820.4,
                                       mass_tol_ppm=20)
```

## spectrum/pfb_reader.py

PFB（pFind/pXtract 二进制谱图）格式的纯解析模块（无 numpy 数组构建 / DIAData 知识）。被 `DIAData._load_from_pfb` 使用。

### 模块常量
- `HEADER_SIZE = 24`（`struct "<iiiqi"`：3 空 int32 + int64 addr_list_addr + int32 scan_num）
- `_MS1_FIELD_COUNT = 4`、`_MS2_FIELD_COUNT = 13`

### `@dataclass PFBSpectrum`
单谱记录：`scan: int`、`ms_level: int`、`rt: float`（**秒**）、`instrument_type: str`、`mz: np.ndarray`、`intensity: np.ndarray`；MS2 专有（MS1 为 None）：`charge, mh_plus, ion_injection_time, activation_center, activation_type, precursor_scan, activation_window, nce, monoisotopic_mz`。

### `read_header(fh) -> tuple[int, int]`
读 24 字节头，返回 `(addr_list_addr, scan_num)`，文件停在首谱（偏移 24）。截断 **Raises** `ValueError`。

### `parse_property_str(s) -> dict`
`\t` 分隔属性串 → 类型化字段；按 `token[1]`（MsType）分派 MS1=4 / MS2=13 字段；字段数不符或未知 MsType **Raises** `ValueError`。

### `iter_spectra(fh, scan_num) -> Iterator[PFBSpectrum]`
顺序读 loop body（`fh` 需先 `read_header`）。每谱：`int32 len` + property_str（`rstrip('\x00')`）+ `int32 peak_num` + `peak_num` 个 double mz + `peak_num` 个 double intensity。负 peak_num / 截断 **Raises** `ValueError`。

### `iter_scan_ids(fh, scan_num) -> Iterator[int]`
pass-1：仅取每谱 scan 号、`seek(peak_num*16)` 跳过峰（不解码），供定长数组 sizing。负 peak_num / 截断 **Raises** `ValueError`。

### `read_footer(fh, addr_list_addr, scan_num) -> list[int]`
读尾部 `scan_num` 个 int64 逐谱偏移（校验用；加载路径不依赖）。截断 **Raises** `ValueError`。

## spectrum/psm_info.py

### `spectrum/labeling.py`

- 模块常量：`MASS_DELTA_C13_C12=1.003355`、`MASS_DELTA_N15_N14=0.997035`、`IDEAL_FULL_LABEL_ISOTOPE_MODEL="ideal_full_label_v1"`。
- `get_fixed_heavy_atom_counts(sequence, heavy_type) -> dict[str,int]` — 返回理想完全标记下不再参与残余天然同位素包络的固定重原子：SILAC 为 K/R 标记 C/N，C13 为全部 C，N15 为全部 N。

### `spectrum/psm_info.py`

模块常量 `PROTON_MASS=1.00727646677`；模块加载时读 `unimod.xml` 为全局 `unimods`。

### `class HeavyType(Enum)`
规范成员为 `SILAC=1`、`C13=2`、`N15=3`。旧源码名 `CHEAVY` / `NHEAVY` 仅作为兼容别名保留；序列化名称始终为 `silac` / `c13` / `n15`。

### `class PSMInfo`
`__init__(sequence, charge, modify, rt, precursor_mz, raw_title, protein_names, q_value=None, score=None, label_type=None)`，`modify` 为 `list[(0基位置, unimod_id)]`。

- `to_dict() -> dict` / `classmethod from_dict(data) -> PSMInfo` — JSON 互转（新字段 None 兜底）。
- `get_key() -> (sequence, charge, ((pos,mod),...))` — 去重键。
- `get_key_with_raw() -> (..., raw_title)` — 含 raw 的去重键。
- `valid() -> bool` — 含 `X` 即 False。
- `get_modify_mass(end_idx) -> float` — `[0, end_idx]` 内修饰单同位素质量之和。
- `_assert_heavy_supported(heavy_type) -> None` — 守卫：`heavy_type∈{C13,N15}` 且 `_modify` 非空时抛 `NotImplementedError`（修饰原子的来源/引入时机未知，避免静默错误质量）。
- `get_SILAC_precursor_mz() -> float` / `get_uniform_label_precursor_mz(heavy_type) -> float` — 重标前体 m/z（后者带修饰 + C13/N15 会抛 `NotImplementedError`）；`get_C_N_HEAVY_precursor_mz` 是旧兼容别名。
- `get_fragment_ions(heavy_type) -> (b_ans, y_ans)` — 元素为 `("b"/"y", 序号, light_mass, heavy_mass)`（带修饰 + C13/N15 会抛 `NotImplementedError`）。
- `get_heavy_info(heavy_type) -> (heavy_precursor_mz, b_ions + y_ions)`。
- `get_theoretical_isotope_ratios(sequence, modifications=(), heavy_type=SILAC) -> list[float]` — 在 `ideal_full_label_v1` 假设下返回归一化 `[M0,M1,M2]`；C13/N15 带修饰时抛 `NotImplementedError`。

### `get_SILAC_increase_mass(sequence) -> float`
K +8.014204、R +10.008275。

### `get_heavy_increase_mass(sequence, heavy_type) -> float`
SILAC 委托上者；C13=`C数×1.003355`；N15=`N数×0.997035`。**仅统计序列原子，不含修饰原子**——带修饰肽段的 C13/N15 已由工作流策略过滤，底层仍由 `_assert_heavy_supported` 拦截。

### `get_theoretical_isotope_ratios(sequence) -> list`
返回 `[1.0, λ, λ²/2]`（Poisson 近似 M0/M1/M2）。

### `has_label_site(sequence, heavy_type=HeavyType.SILAC) -> bool`
该肽段在 `heavy_type` 下是否存在代谢标记位点（即轻/重 SILAC 式校验是否有意义）。SILAC 只标记 K/R——无 K/R 的肽段没有重标搭档（重=轻），返回 False；CHEAVY(¹³C)/NHEAVY(¹⁵N) 为全原子代谢标记，任何肽段都含 C 和 N，故恒为 True。空序列返回 False。大小写不敏感（内部 `upper()`）。

### `sequence_controlled_shuffle(peptide, anchor_len=2, shuffle_ratio=0.5, seed=None, max_tries=10) -> str`
保留 C 端 `anchor_len` 残基，核心区按比例打乱；`seed` 给定则可复现。

## spectrum/pfind_parser.py

### 模块常量
`PROTON_MASS=1.00727646677`、`PFIND_MOD_TO_UNIMOD`（pFind 修饰名→UniMod ID 硬编码）、`PFIND_DECOY_PREFIX="REV_"`。

### `resolve_pfind_mod_name(name) -> int | None`
先查硬编码字典，再用 base name 查 UniMod；`lru_cache(1024)`。

### `parse_pfind_modify(modify_str) -> list[tuple[int, int]]`
解析 `"位置,名称;..."`，位置 1基→0基；非 str/空 → `[]`；未知修饰 log warning 跳过。

### `mhp_to_mz(mhp, charge) -> float`
`(mhp + (charge-1)*PROTON_MASS)/charge`；charge≤0 **Raises** `ValueError`。

### `extract_raw_title_from_pfind_path(path) -> str`
去 `.qry.res` 后缀取 basename。

### `load_pfind_file(file_path, qvalue_threshold=0.01) -> list[PSMInfo]`
读 `.qry.res`（TSV），过滤 FDR → 全 decoy → invalid；RT=`PredRT+DeltaRT(Min)`。文件不存在返回 `[]`。

### `load_pfind_path(path, qvalue_threshold=0.01) -> list[PSMInfo]`
目录则扫描所有 `*.qry.res`（排序）逐个加载并合并；单文件则只加载该文件。

## spectrum/light_result.py

### `class LightResult`
`__init__()`：`peptide_len=0`、`psm_info=[]`。

- `_load_from_pkl(path)` — 读自定义 JSON，`PSMInfo.from_dict` 重建。
- `_load_from_alphadia_input(path, qvalue_threshold=0.01)` — 读 `precursors.parquet`；缺列 **Raises** `ValueError`。
- `_load_from_dia_nn_input(path, qvalue_threshold=0.01)` — 读 DIA-NN `report.parquet`；缺列 **Raises** `ValueError`。
- `_load_from_pfind_input(path, qvalue_threshold=0.01)` — 委托 `load_pfind_path`。
- `filtered_by_raw_title(raw_title) -> np.ndarray[PSMInfo]`。

### `parse_diann_peptide_modify(sequence) -> list[(int, int)]`
解析 DIA-NN 内联 `(...UniMod:n...)`，括号不计残基；非 UniMod 跳过。

### `parse_alphadia_peptide_modify(modify_str, site_str) -> list[(int, int)]`
`;` 分隔的 mods/sites 等长配对，位置 1基→0基；未知修饰/坏位置跳过。

### `rt_sec_to_min(rt) -> float`
`rt / 60`。

## spectrum/entrapment_classifier.py

### `@dataclass TargetIndex`
字段：`raw_text: str`、`li_normalized_text: str`、`n_proteins: int`。

### `load_target_fasta(fasta_path, log_label="target FASTA") -> TargetIndex`
读 FASTA 拼接（分隔符 `|`）+ `I→L` 归一化；不存在 **Raises** `FileNotFoundError`。`log_label` 控制日志（`加载 {log_label}: ...`）与报错措辞——entrapment 用默认 "target FASTA"，污染库过滤传 "污染库"，避免把污染库误称为 target 蛋白组。

### `classify_peptide(peptide, target) -> str`
返回 `"L0"`（精确子串）/`"L1"`（L↔I 归一后子串）/`"L4"`（都不是）；空肽 → `"L4"`。

### `classify_peptides_batch(peptides, target) -> list[str]`
按序批量分级。

## spectrum/species_marker.py

### `matches_species_marker(protein_names: str | None, marker: str) -> bool`
任一非 decoy token 以 `_<marker>` 结尾（或等于 marker）则 True；空/None 输入或空 marker → False。UniProt `sp|P12345|GENE_HUMAN` 取末段比较。

示例：
```python
matches_species_marker("sp|P12345|ALBU_HUMAN", "HUMAN")  # True
matches_species_marker("REV_X_HUMAN", "HUMAN")           # False (decoy)
```

## spectrum/spectrum_utils.py

### `match_peak_ppm(mz_arr, intensity_arr, precursor_mz, mass_tol_ppm) -> (np.float32, np.float32)`
ppm 容差内峰的 (平均 ppm 误差, 强度之和)；无匹配返回 `(nan, 0.0)`。

### `centroid_spectrum(mz, intensity, rel_threshold=1e-3) -> (np.ndarray, np.ndarray)`
profile 谱质心化。**参数**：`mz` 单调递增 m/z；`intensity` 同长强度；`rel_threshold` 相对峰高阈值。**返回**：等长 (mz_out, intensity_out)，dtype 跟随输入；长度<3 或无峰返回空数组。**Raises** `ValueError`（mz 与 intensity 长度不等）。
