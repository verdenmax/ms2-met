# spectrum — 职责与接口

## 一句话职责

承载 **DIA 原始谱图（mzML/PFB/npz）的加载与 XIC 提取**、**PSM 信息与 SILAC/重标质量计算**、以及各搜索引擎（pFind / DIA-NN / Alphadia）结果解析与物种/entrapment 标注，为上层 `workflows` 的轻重标配对验证提供数据底座。

> 本文档不含已单独成册的 `speclib` 子包。

## 对外接口

### `dia_data.py` — DIA 数据与 XIC

| 符号 | 签名 | 简述 |
|---|---|---|
| `DIAData` | class | 一个 raw 的全部谱图（mz/intensity/rt + DIA 窗口 + 索引），扁平数组存储 |
| `DIAData.load_from_file` | `(filepath, use_mmap=True, expected_centroid_*) → DIAData` | 从 `.npz` 缓存加载（mmap 零拷贝），校验 `_format_version=3` 与 centroid 参数 |
| `DIAData.save_to_file` | `(filepath)` | 存为 `.npz`（`savez_compressed`）|
| `DIAData.validate_cache_params` | `(filepath, enabled, rel_threshold)` | 仅读 3 个标量做轻量缓存校验 |
| `DIAData.xic_ms2_peaks_extract` | `(rt, xic_cycle_window, precursor_mz, ions_mass, mass_tol_ppm) → (结构化ndarray, total_intensity)` | MS2 层 XIC：按 DIA 窗口逐 cycle 提取碎片离子色谱 |
| `DIAData._load_from_mzml` / `_load_from_pfb` | `(path)` | 从 mzML / PFB 加载（两遍）；`DataManager.get_dia_data_object` 按扩展名分派，二者产出等价 DIAData |
| `DIAData.xic_peaks_extreact` | `(rt, xic_cycle_window, precursor_mz, mass_tol_ppm) → ndarray` | MS1 层 XIC：前体离子色谱 |
| `DIAData.get_window_info` / `check_in_raw` / `check_in_same_ms2` | — | DIA 隔离窗口查询 / 范围判断 |
| `deduplicate_with_tolerance` | `(arr, tolerance=0.1) → ndarray` | float32 容差去重并排序（构造 DIA 窗口左边界）|

### `pfb_reader.py` — PFB 二进制谱图解析（纯解析，无 DIAData 知识）

| 符号 | 签名 | 简述 |
|---|---|---|
| `PFBSpectrum` | dataclass | 一张谱图：scan / ms_level / rt(秒) / instrument_type / mz / intensity（+ MS2 专有字段）|
| `read_header` | `(fh) → (addr_list_addr, scan_num)` | 读 24 字节头，定位到首谱 |
| `parse_property_str` | `(s) → dict` | `\t` 分隔属性串 → 类型化字段（按 MsType：MS1=4 / MS2=13）|
| `iter_spectra` | `(fh, scan_num) → Iterator[PFBSpectrum]` | 顺序读 loop body（intensity 为 double）|
| `iter_scan_ids` | `(fh, scan_num) → Iterator[int]` | pass-1 仅取 scan 号、跳过峰（用于定长数组 sizing）|
| `read_footer` | `(fh, addr_list_addr, scan_num) → list[int]` | 读尾部逐谱偏移（校验用，加载路径不依赖）|

> 详见 `docs/specs/2026-06-12-pfb-format-support-design.md`。

### `psm_info.py` — PSM 与重标质量

| 符号 | 签名 | 简述 |
|---|---|---|
| `PSMInfo` | class | 一个 PSM（序列/电荷/修饰/RT/前体 m/z/...），可 `to_dict`/`from_dict` |
| `HeavyType` | Enum | `SILAC` / `CHEAVY`(¹³C) / `NHEAVY`(¹⁵N) |
| `PSMInfo.get_heavy_info` | `(heavy_type) → (heavy_precursor_mz, ions)` | 重标前体 m/z + b/y 离子（轻+重质量）|
| `PSMInfo.get_fragment_ions` | `(heavy_type) → (b_ions, y_ions)` | 含修饰质量的理论碎片离子 |
| `PSMInfo.get_key` / `get_key_with_raw` | — | PSM 去重键（序列+电荷+修饰[+raw]）|
| `get_SILAC_increase_mass` | `(sequence) → float` | K +8.014204、R +10.008275 |
| `get_heavy_increase_mass` | `(sequence, heavy_type) → float` | 按重标类型计算质量增量 |
| `get_theoretical_isotope_ratios` | `(sequence) → [M0,M1,M2]` | Poisson 近似同位素比例 |
| `sequence_controlled_shuffle` | `(peptide, anchor_len=2, shuffle_ratio=0.5, seed=None) → str` | 锚定 C 端的可控乱序（生成 entrapment）|

### `pfind_parser.py` — pFind 结果解析

| 符号 | 签名 | 简述 |
|---|---|---|
| `load_pfind_path` | `(path, qvalue_threshold=0.01) → list[PSMInfo]` | 目录扫描 `*.qry.res` 或单文件 |
| `load_pfind_file` | `(file_path, qvalue_threshold=0.01) → list[PSMInfo]` | 单文件 + FDR/decoy/valid 过滤 |
| `parse_pfind_modify` | `(modify_str) → list[(0基位置, unimod_id)]` | 解析 Modifications 字段 |
| `resolve_pfind_mod_name` | `(name) → int\|None` | pFind 修饰名 → UniMod ID |
| `mhp_to_mz` | `(mhp, charge) → float` | MH⁺ → m/z |
| `extract_raw_title_from_pfind_path` | `(path) → str` | 去 `.qry.res` 后缀取 raw 名 |

### `light_result.py` — 多引擎轻标结果容器

| 符号 | 签名 | 简述 |
|---|---|---|
| `LightResult` | class | 统一持有 `list[PSMInfo]`，支持 pkl(json)/Alphadia/DIA-NN/pFind 四种来源 |
| `LightResult._load_from_pfind_input` / `_load_from_dia_nn_input` / `_load_from_alphadia_input` / `_load_from_pkl` | — | 各来源加载（FDR+decoy 过滤）|
| `LightResult.filtered_by_raw_title` | `(raw_title) → ndarray[PSMInfo]` | 按 raw 过滤 |
| `parse_diann_peptide_modify` | `(sequence) → list[(pos, unimod_id)]` | DIA-NN `(UniMod:n)` 内联修饰解析 |
| `parse_alphadia_peptide_modify` | `(modify_str, site_str) → list[(pos, unimod_id)]` | Alphadia mods/sites 解析 |
| `rt_sec_to_min` | `(rt) → float` | 秒→分 |

### `entrapment_classifier.py` — 陷阱肽分级

| 符号 | 签名 | 简述 |
|---|---|---|
| `TargetIndex` | dataclass | 拼接 target proteome 文本 + L↔I 归一化视图 |
| `load_target_fasta` | `(fasta_path) → TargetIndex` | 读 FASTA 拼接成可子串搜索的文本 |
| `classify_peptide` | `(peptide, target) → "L0"\|"L1"\|"L4"` | 单肽是否与 target 质谱不可区分 |
| `classify_peptides_batch` | `(peptides, target) → list[str]` | 批量分级 |

### `species_marker.py` — 物种标记匹配

| 符号 | 签名 | 简述 |
|---|---|---|
| `matches_species_marker` | `(protein_names, marker) → bool` | 蛋白名是否含 `_<marker>` 后缀且非 decoy |

### `spectrum_utils.py` — 谱图基础算子

| 符号 | 签名 | 简述 |
|---|---|---|
| `match_peak_ppm` | `(mz_arr, intensity_arr, target_mz, mass_tol_ppm) → (ppm_error, total_intensity)` | ppm 容差内峰匹配 |
| `centroid_spectrum` | `(mz, intensity, rel_threshold=1e-3) → (mz_out, int_out)` | profile 谱图质心化（抛物线插值）|

## 依赖

- 第三方库：`numpy`、`pandas`、`pyteomics`（`mzml` 读取、`mass.Unimod`/`fast_mass`/`Composition`）。
- 内部依赖：`light_result`/`pfind_parser` → `psm_info`；`dia_data` → `spectrum_utils`；均不依赖 `speclib`。
- 数据文件：仓库根 `unimod.xml`（修饰单同位素质量）、target FASTA、各引擎报告（`.qry.res` / `report.parquet` / `precursors.parquet`）。
- 被依赖：`manager/data_manager.py`（DIAData 缓存）、`manager/light_result_manager.py`、`workflows/{flow_utils,pair_flow,single_work}.py`、`tools/{extract_common,eval_baseline,entrapment_classify}.py`。

## 输入 / 输出

- 输入：mzML / npz 缓存、各搜索引擎结果文件、target FASTA、`unimod.xml`。
- 输出：`DIAData`（谱图与 XIC ndarray）、`list[PSMInfo]` / `LightResult`、重标质量与碎片离子、物种/entrapment 标签。
