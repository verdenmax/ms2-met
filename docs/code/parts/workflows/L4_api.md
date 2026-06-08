# workflows — API 参考

逐文件列出主要 class/函数。签名以实际代码为准。

## workflows/pair_flow.py

### `class PairFlow`

顶层工作流，`main.py` 入口。模块常量 `BATCH_SIZE = 5000`。

- `__init__(self, workname: str, config: ConfigParser | None = None, work_path: str = "./Pairworkspace")` — 记录配置、建工作目录。类常量 `RAW_DATA_MANAGER_PICKLE`、`LIGHT_RESULT_MANAGER_PUCKEL`。
- `load(self) -> None` — 构造 `LightResultManager` 读 `light_result_file` 得 `psm_info`；构造并 `save()` `DataManager`。
- `run(self) -> None` — `load()` 后 `distribute()`。
- `distribute(self) -> None` — 两阶段：① 进程池 `data_to_npz` 生成 DIA npz 缓存；② 按 `feature_type`(0/1/2) 生成任务、按 shared 路径分桶切 `BATCH_SIZE`，进程池跑批函数，`as_completed` 汇总写 `result_file`；崩溃写 `*.PARTIAL_INCOMPLETE`。
- `multi_handle(self, psm1: PSMInfo, psm2: PSMInfo, label: int) -> dict` — 单进程路径：调 `multi_batch_work`，返回元数据+特征+`label` 行。
- `pharse_data(self, tot_raw_path: str) -> tuple[str, DIAData]` — 经 `DataManager` 取 DIA 对象。
- `_process_group(self, group) -> list[dict]` — 组内两两组合生成正例与（heavy `_rt+10` 的）负例。

## workflows/flow_utils.py

- `get_filename_stem(filepath: str) -> str` — 去目录与扩展名。
- `data_to_npz(raw_file_manager: DataManager, filepath: str, _workpath: str = ".") -> tuple[str, str]` — 校验/重建 `<name>.dia.npz` 缓存，返回 `(name, shared_path)`。
- `process_psm_pair_shared(psm1_dict, psm2_dict, shared1_file, shared2_file, config, label) -> dict` — mmap 加载两 DIA，`label==0` 时 heavy `_rt+=10`，调 `multi_batch_work`，返回结果行。
- `process_psm_single(psm1_dict, shared1_file, config) -> dict` — mmap 加载单 DIA，调 `single_pair_work`，经 `_make_result_row_single` 返回行。
- `_make_result_row_single(psm, features: dict) -> dict` — 把 `psm._label_type` 映射成 1/0；为 `None` 抛 `ValueError`。
- `process_batch_single(shared_path: str, batch_psm_dicts: list, config) -> tuple[list, int]` — 批处理 feature_type 0；逐 PSM 捕获异常计数。
- `process_batch_pair(shared1, shared2, batch_items: list, config) -> tuple[list, int]` — 批处理 feature_type 1（负例 heavy `_rt+=10`）。
- `process_batch_pair_shuffle(shared1, shared2, batch_items: list, config) -> tuple[list, int]` — 批处理 feature_type 2（负例 `sequence_controlled_shuffle`，种子 = `random_seed` + `crc32(seq)`）。

> 批函数返回 `(results, n_errors)`；`results` 为成功行列表，`n_errors` 为被捕获并记录的逐 PSM 异常数。

## workflows/single_work.py

- `multi_batch_work(psm1: PSMInfo, dia_data1: DIAData, psm2: PSMInfo, dia_data2: DIAData, config: ConfigParser) -> dict` — 双文件/双 PSM 特征提取，返回完整特征 dict。
- `single_pair_work(psm: PSMInfo, dia_data: DIAData, config: ConfigParser) -> dict` — 单文件特征提取（内部经 `get_heavy_info(HeavyType.SILAC)` 推 heavy），schema 与 `multi_batch_work` 对齐。
- `calc_xic_score(light_xic, heavy_xic, center_rt: float|None=None, heavy_center_rt: float|None=None, intensity_threshold: float=1e-10) -> dict` — 一对 XIC 的 19 字段打分（pearson/cosine/apex_delta(±)/mz_avg_err/强度比/snr/峰形/cycle_offset…）；空对返回 `_default_xic_score()`。
- `extract_ion_pearson_features(ions_pearsons: list) -> dict` — count/p25/p50/p75/mean/std/min/high_ratio；`count==1` 时 std=NaN。
- `extract_ion_numeric_features(values: list, prefix: str) -> dict` — `{prefix}_mean/p50/std/max`，清洗 NaN/Inf。
- `_is_empty_xic_pair(light_xic, heavy_xic) -> bool` — 任一 XIC 空或全零强度。
- `_calc_fwhm(rt, intensity) -> float`、`_calc_symmetry(intensity) -> float`、`_calc_snr(intensity) -> float`、`_calc_base_to_apex_ratio(intensity) -> float`、`_calc_apex_monotonicity(intensity) -> float`、`_calc_n_peaks(intensity, prominence_frac=0.3) -> int`、`_calc_smoothness(intensity) -> float` — 单 XIC 峰形指标。
- `_calc_cycle_offset(xic, center_rt: float) -> tuple[int, int]` — apex 相对 center_rt 的 (abs, signed) 周期偏移。
- `_calc_hl_ratio_consistency(ratios: list) -> tuple[float, float]` — log10(L/H) 的 (std, mad)；`count==1` 时 std=NaN。
- `_default_xic_score() -> dict` — calc_xic_score 全零默认返回。
- `plot_light_heavy_contract(ion_data)`、`plot_light_heavy_xic(light_xic, heavy_xic)` — 可选 matplotlib 绘图辅助（运行期非必需）。

## workflows/q1a_helpers.py

实现 spec §4.2 的碎片配对召回。模块常量 `DEFAULT_INTENSITY_FLOOR=100.0`、`DEFAULT_APEX_DELTA_FRACTION=0.3`、`DEFAULT_PEARSON_MIN=0.5`、`SHIFT_EPSILON=0.001`。

- `is_signal_present_light(xic, intensity_floor: float=DEFAULT_INTENSITY_FLOOR) -> bool` — XIC 峰强 > floor。
- `is_signal_present_heavy(light_xic, heavy_xic, intensity_floor=..., apex_delta_fraction=..., pearson_min=...) -> bool` — 三条件：强度过阈 + apex RT 差 < `apex_delta_fraction * light_peak_width` + 共网格 Pearson > `pearson_min`。
- `is_split_window(w_light: dict, w_heavy: dict) -> bool | None` — 窗口不同 True / 相同 False / 任一边界 NaN 时 None。
- `is_separable_fragment(light_mass: float, heavy_mass: float, split_window, shift_epsilon: float=SHIFT_EPSILON) -> bool` — 被 SILAC 位移，或窗口已知分裂。
- `class Q1aAccumulator` — 逐 PSM 累计器。常量 `MIN_VALID_TOTAL=3`、`VALID_ION_TYPES=("b","y")`。
  - `__init__(self, split_window: bool, intensity_floor=..., apex_delta_fraction=..., pearson_min=...)`
  - `add(self, ion_type: str, light_mass: float, heavy_mass: float, light_xic, heavy_xic) -> None` — 累计一个理论碎片；`ion_type` 非 b/y 抛 `ValueError`；非有限质量或负位移发 `RuntimeWarning` 并跳过；不可分离或无 light 信号静默丢弃。
  - `compute_features(self) -> dict` — 返回 11 字段 q1a 特征（recall 系列在桶 < 3 时 NaN，count 系列恒整数）。
