# spectrum — 细节

## DIA 数据存储（`dia_data.py`）

### 数据结构（扁平 + 索引）

- 所有谱图的峰拼成两条全局一维数组 `_mz_values` / `_intensity_values`（float32）；每个谱图用 `_peak_start_idx_list[i] : _peak_stop_idx_list[i]` 切片定位（`get_spectrum_by_index`）。
- 按谱图数定长的数组：`rt_values`（分钟）、`precursor_scan_ids`（MS1=-1）、`_precursor_lower_mz` / `_precursor_upper_mz`（隔离窗口绝对边界）。
- `_scan_id_to_index`：scan number → 谱图下标的反查表，按 `max(scan_id)+1` 而非谱图数 sizing——pParse/ProteoWizard 过滤后 scan 号可远大于谱图数，否则会 IndexError。
- `ms1_indexs` / `ms2_indexs`：由 `precursor_scan_ids == -1` 与 `!= -1` 派生；对应 `*_rt` 为各自 RT 子数组。
- `_cycle_left_precursor`：对所有窗口下边界做容差去重（`deduplicate_with_tolerance`，tol=0.1）得到的 DIA 窗口左边界集合，用于判断两个前体是否落在同一 MS2 窗口。

### mzML 两遍加载

- 第一遍只迭代 `spectrum['id']` 统计 `total_spectra` 与 `max_scan_id`（pyteomics 懒解码，不读峰数组）。
- 第二遍逐谱处理：`_process_single_spectrum` 返回 (mz_chunk, int_chunk)，累积到 list 后一次 `np.concatenate`。峰值内存约为最终数组 2×（concat 时同时持有 chunk list 与新数组）。
- scan number 从 id 串正则 `scan=(\d+)` 提取；提取不到则抛 `ValueError`。
- RT 单位处理：读 `scan start time` 的 `unit_info`，`second` → /60 转分钟，`minute`/无单位按分钟，其它单位抛错；无字段返回 0.0。
- MS2 前体：取 `precursorList.precursor[0]`，隔离窗口由 `selected ion m/z ± isolation window offset` 换算为绝对 `lower/upper`。

### 加载时质心化（centroiding）

- `_centroid_enabled`（默认 True）+ `_centroid_rel_threshold`（默认 1e-3）。若谱图已带 CV term `centroid spectrum`（`_is_already_centroid`）则跳过。
- 质心化后空返回（<3 峰或全零）计入 `_n_centroid_empty`，加载末尾汇总 log。

### PFB 两遍加载（`_load_from_pfb` + `pfb_reader.py`）

- **格式分派**：`DataManager.get_dia_data_object` 按扩展名选择 `.pfb` → `_load_from_pfb`，否则 `_load_from_mzml`；二者经共享的 `_record_spectrum`（逐谱写定长数组）与 `_finalize_arrays`（concat + 派生 ms1/ms2 索引、mz 范围、DIA 窗口左界）产出**等价** DIAData。
- **二进制布局**（小端，实测验证）：24 字节头（3 空 int + int64 addr_list_addr + int32 scan_num）；loop body 每谱 `int32 len + UTF-8 property_str + int32 peak_num + double[] mz + double[] intensity`；尾部 `int64[scan_num]` 偏移。**intensity 是 double**（非 float）。
- **两遍**：pass-1 `iter_scan_ids` 只取 scan 号、`seek` 跳过峰，得 `scan_num` 与 `max_scan_id` 给 `_preallocate_arrays`；pass-2 `iter_spectra` 解码峰并填充。
- **RT 单位**：PFB 的 RT 是**秒**，管线规范是**分钟**（与 `_get_retention_time` 一致），故 `_load_from_pfb` 逐谱 `/60` 转换——否则 XIC 的 `searchsorted(rt)` 会因 60× 偏差全部落空。
- **字段映射**：MS1 → `precursor_scan_id=-1`、隔离窗口 `None`（存为 NaN）；MS2 → `precursor_scan_id=precursor_scan`、隔离窗口 `activation_center ± activation_window/2`。MS1/MS2 划分仍由 `precursor_scan_ids == -1` 派生，与 mzML 同源。
- **不质心化**：PFB（pXtract 导出）已是 peak-picked，`_load_from_pfb` 不调用 `centroid_spectrum`（config 的 centroid 设置对该路径无效，加载时 debug 提示一次）。
- **错误处理**：截断 / 负长度 / property token 数不符 MsType / MS2 缺 ActivationWindow / 未知 MsType 均抛 `ValueError`（带谱序号、偏移、scan 等上下文）；空文件（scan_num=0）产出空 DIAData。

### npz 缓存与版本

- `_format_version=3`：相对 v2 新增「内嵌 centroid 参数」。加载/校验时若版本≠3 或 centroid 参数与配置不符则抛 `ValueError`，强制重建缓存（避免用旧 profile 缓存）。
- `validate_cache_params` 用 `with np.load(..., mmap_mode='r')` 只读少量元数据标量（`_format_version`、`_centroid_enabled`、`_centroid_rel_threshold`，以及给定 `expected_source_path` 时的 `_source_size`/`_source_mtime`）再关闭句柄——避免为校验元数据而 mmap 数 GB 数组，并防止 Windows 删除文件时的句柄竞争。
- **源文件身份失效**：`save_to_file(source_path=...)` 把源 mzML/PFB 的 `mtime`/`size` 写入缓存；`validate_cache_params(expected_source_path=...)` 比对，size/mtime 不符则抛 `ValueError` 触发重建。**跳过 size/mtime 比对的两种情形（旧缓存无 `_source_size` 字段、或源文件不存在）现在会发出明确 `WARNING`**（提示可能复用与当前输入不一致的缓存、建议删除重建 / 确认 raw_path），不再静默命中。`workflows/flow_utils.py` 命中前即用它校验；命中日志为「命中（params 校验通过）」（不再谎称"源文件匹配"）。
- `save_to_file` 过滤掉 None 值（`np.savez` 不支持 None）、原子写（同目录 `mkstemp` + `os.replace`）；加载用 `_load_attrs` 填充，`_format_version` 故意不回填到对象。

### XIC 提取逻辑

**MS2 层 `xic_ms2_peaks_extract`**（核心验证算法）：
1. `searchsorted(ms2_indexs_rt, rt)` 定位 `pos`，再从 `pos` **向左（含 pos）无界扫描**、从 `pos+1` **向右无界扫描**，各取第一个「`precursor_mz` 落在隔离窗口 `[lower-0.1, upper+0.1]` 内」的 MS2 谱图，两侧命中里取 RT 更接近者作为 center；全程无匹配则记 `_n_out_of_window_xic` 并返回空。（DIA 每 cycle 轮询 N 个窗口，含该前体的窗口每 N 个 MS2 才出现一次，N 常 20–70，故不能用固定 ±k 候选。）
2. 从 center 向左/右各收集 `xic_cycle_window` 个「有效」MS2 谱图，跳过 NaN 窗口。注意：收集阶段用严格判定 `lower <= precursor_mz <= upper`（无 ±0.1 容差），与 center 选择阶段的 `[lower-0.1, upper+0.1]` 略有不同。
3. 对每个选中谱图，对 charge=1,2 算理论 m/z `(ions_mass + z·proton)/z`，`match_peak_ppm` 在 ppm 容差内累加强度、累加 ppm 误差。
4. 返回结构化 ndarray `[(rt, ppm_error, intensity, cycle_idx)]` + 全窗口 total_intensity。`cycle_idx` 由 `_ms2_cycle_idx` 经 precursor_scan_id → MS1 全局下标 → 在 `ms1_indexs` 中的位置得到。

**MS1 层 `xic_peaks_extreact`**：以 `find_near_ms1_idx`（最近 MS1）为中心，前后各 `xic_cycle_window` 个 MS1，对 precursor_mz 做 `match_peak_ppm`。

### 坑

- `proton_mass=1.00727646677` 在多处硬编码（dia_data/psm_info/pfind_parser 各一份）。
- `mobility_values`/`_quad_*` 等字段标注 TODO/未使用。
- `get_spectrum_by_rt` 假设 RT 单调递增，直接 `searchsorted` 不做最近匹配。

## PSM 与重标质量（`psm_info.py`）

### 修饰与碎片离子

- `modify` 为 `list[(0基位置, unimod_id)]`。`get_modify_mass(end_idx)` 累加 ≤end_idx 的修饰单同位素质量（来自全局 `unimod.xml` 的 `mass.Unimod`）。
- `get_fragment_ions` 用 `mass.fast_mass(subseq, ion_type='b'/'y')`，b 离子加前缀修饰质量、y 离子加 (全修饰 − 前 n-i-1 修饰)；同时给出重标质量 `+ get_heavy_increase_mass`。返回 `("b"/"y", 序号, light_mass, heavy_mass)`。

### SILAC / C/N 重标质量

- SILAC：每个 K +8.014204（C(-6)¹³C(6)N(-2)¹⁵N(2)）、R +10.008275（C(-6)N(-4)¹³C(6)¹⁵N(4)），硬编码常量。
- CHEAVY：`Composition(seq)['C'] × (¹³C−¹²C=1.003355)`；NHEAVY：`['N'] × (¹⁵N−¹⁴N=0.997035)`。
- 重标前体 m/z = (轻前体质量 + 质量增量)/charge。
- **⚠️ 修饰原子的重标未实现**：CHEAVY/NHEAVY 全代谢标记下，修饰基团里的 C/N 原子同样应被 ¹³C/¹⁵N 替换，但 `get_heavy_increase_mass` 只统计序列骨架/侧链原子。为避免静默返回错误质量，`get_C_N_HEAVY_precursor_mz` / `get_fragment_ions` 在 `heavy_type∈{CHEAVY,NHEAVY}` 且肽段带修饰（`_modify` 非空）时经 `_assert_heavy_supported` 抛 `NotImplementedError`（代码内 TODO）。SILAC 只标记 K/R、不涉及修饰，**不受影响**；无修饰的 CHEAVY/NHEAVY 仍正确。
- `has_label_site(sequence, heavy_type)`：判断肽段是否存在标记位点。SILAC 仅当含 K/R 才有重标搭档（否则重=轻、不可校验）；CHEAVY/NHEAVY 为全原子标记，任何非空肽段都含 C/N 故恒 True；空序列 False。上游 `tools/extract_common`、`tools/trap_domain_filter` 用它筛掉无法做轻重校验的肽段。

### 同位素比例与乱序

- `get_theoretical_isotope_ratios`：Poisson 近似，λ 由各元素重同位素天然丰度加权（¹³C 0.01109、²H 0.000115、¹⁵N 0.00364、¹⁷O 0.00038、³³S 0.0079），返回 `[1, λ, λ²/2]`。
- `sequence_controlled_shuffle`：保留 C 端 `anchor_len`（默认 2，保 y1/y2 离子）个残基，仅对核心区 `shuffle_ratio` 比例打乱；`seed` 给定时用独立 `random.Random(seed)` 保证可复现。

## 字段解析规则

### pFind（`pfind_parser.py`）

- `parse_pfind_modify`：输入 `"3,Carbamidomethyl[C];10,...;"`，位置 **1基→0基**，非字符串/NaN/空 → `[]`；逐项 `split(",",1)`，未知修饰 log warning 跳过。
- 修饰名解析 `resolve_pfind_mod_name`：先查硬编码 `PFIND_MOD_TO_UNIMOD`，未命中取 `[` 前 base name 查 UniMod `by_title`，仍无则 None；`lru_cache(1024)`。
- `mhp_to_mz`：`(MH⁺ + (z-1)·proton)/z`，charge≤0 抛 `ValueError`。
- `load_pfind_file` 过滤顺序：QValue>阈值（FDR）→ 全部蛋白 token 都是 `REV_` 前缀才算 decoy（混合命中保留）→ `PSMInfo.valid()`。RT = `PredRT + DeltaRT(Min)`；用 `itertuples` 并把 `MH+`/`DeltaRT(Min)` 重命名为合法标识符。

### DIA-NN / Alphadia（`light_result.py`）

- DIA-NN `parse_diann_peptide_modify`：扫描 `Modified.Sequence` 内联 `(...UniMod:n...)`，括号不计入残基计数得 0 基位置；非 UniMod 修饰 log warning 跳过。无 `Stripped.Sequence` 时用 `re.sub(r"\(.*?\)","")` 剥离。decoy 判定：`Decoy!=0` 或蛋白名以 `REV_`/`_REV`/`DECOY_` 开头。
- Alphadia `parse_alphadia_peptide_modify`：`mods`/`mod_sites` 各以 `;` 分隔需等长，修饰名经内部小字典映射 UniMod，位置 1基→0基；RT 由 `rt_sec_to_min` 秒转分。decoy 判定 `precursor.decoy!=0`。
- 三种来源均做「FDR → decoy → valid」三段过滤并统计 total/kept/各类丢弃数。

## entrapment 分级（`entrapment_classifier.py`）

- 目的：判断陷阱肽是否与 target proteome 质谱不可区分。`load_target_fasta` 把所有蛋白序列用分隔符 `|`（标准 FASTA 不出现，防跨蛋白边界误匹配）拼成 `raw_text`，并生成 `I→L` 归一化文本。
- `classify_peptide`：`L0`=精确子串命中；`L1`=仅在 L↔I 归一化后命中（L/I 同质量 113.08406）；`L4`=都不是（不在此计算需要 Hamming 扫描的 L2/L3）。空肽返回 L4。
- 用子串匹配而非在体外酶切：更简单、更保守（子串命中是酶切命中的超集），对 SILAC 验证「宁可多清负集噪声」是更安全方向。

## 物种标记（`species_marker.py`）

- `matches_species_marker`：按 `;`/`/` 拆 token；UniProt 式 `sp|P12345|GENE_HUMAN` 取最后一段，判断是否 `endswith("_<marker>")`（marker 大小写敏感）。
- 不用子串匹配：避免 `HUMANIN` 等误命中，且任一 `|` 段以 `REV_`/`DECOY_`/`_REV_`/`_DECOY_` 开头则整 token 视为 decoy 不计。

## 谱图基础算子（`spectrum_utils.py`）

- `match_peak_ppm`：`ppm=(mz-target)/target·1e6`，容差内强度求和、ppm 误差取均值；无匹配返回 `(nan, 0.0)`。
- `centroid_spectrum`：profile 质心化。局部极大判定用非对称规则 `interior > left & interior >= right`（平台取最左内部点）；过 `max·rel_threshold` 阈值；用三点抛物线插值精修 m/z，`|dx|>0.5` 视为拟合失败回退 bin 中心；强度取顶点采样值。长度<3 或无峰返回空数组，dtype 跟随输入。
