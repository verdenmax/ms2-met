# entry — 细节

## 启动流程（`main()`）

1. **参数解析**：`argparse` 定义 `--configpath`（默认 `./config.ini`）与 `--logpath`（默认 `./ms2.log`）。
2. **读取配置**：`configparser.ConfigParser().read(configpath)`。
   - 返回空列表（文件不存在/不可读）→ 抛 `FileNotFoundError`。
   - 读取成功但 `config.sections()` 为空（无任何 `[section]`）→ 抛 `ValueError`。
3. **日志准备**：若 `--logpath` 含父目录，`os.makedirs(log_dir, exist_ok=True)` 确保存在，避免 `FileHandler` 失败。
4. **日志注册**：`logging.basicConfig(level=INFO, handlers=[RichHandler(), file_handler])`。
   - 终端用 `RichHandler()`；文件用 `FileHandler(logpath, encoding="utf-8")`，格式 `%(asctime)s | %(levelname)s | %(name)s | %(message)s`。
5. **开始 banner**：`banner.show_start_banner()`。
6. **取工作目录**：从 `[general]` 读 `work_directory`，缺省 `./workspace`（见下方边界）。
7. **运行工作流**：`PairFlow(workname="main", config=config, work_path=work_path).run()`。
8. **结束 banner**：`banner.show_end_banner()`。

## `config.ini` 结构（`main.py` 直接消费）

| section | option | 用途（来自实际 config.ini） |
|---|---|---|
| `[input]` | `raw_num` | 原始谱图数量 |
| `[input]` | `raw_path_1` / `raw_path_2` | 各原始谱图路径（`.mzML` / `.raw`）|
| `[input]` | `light_result_file` | 轻标结果文件路径 |
| `[input]` | `search_engine_type` | 搜索引擎类型：0=自定义 JSON，1=DIA-NN parquet，2=AlphaDIA parquet，3=pfind `.qry.res` |
| `[input]` | `pfind_qvalue_threshold` | 仅 `search_engine_type=3` 生效，pfind 的 FDR 阈值 |
| `[general]` | `work_directory` | **`main.py` 显式读取**；工作目录，缺省 `./workspace` |
| `[general]` | `feature_type` | 特征生成模式：0=相同文件之间，1=正常轻重标 |
| `[general]` | `mass_tol_ppm` | 质量容差（ppm）|
| `[general]` | `xic_cycle_window` | XIC 周期窗口 |
| `[general]` | `result_file` | 输出结果文件名 |
| `[general]` | `centroid_enabled` | 加载 mzML 时是否对 profile 谱做 centroiding |
| `[general]` | `centroid_rel_threshold` | centroid 相对强度阈值（典型 1e-4~1e-2，推荐 1e-3）|

> 注：`main.py` 仅直接读 `[general] work_directory`，其余 option 在下游模块（如 `PairFlow`、`manager/data_manager.py`）按需读取；`config` 对象整体传入 `PairFlow`。

## `extract_common_config.ini.example` 简述

属于 `tools/extract_common.py`（独立预处理工具，非 `main.py` 消费）的配置示例：用 `[extract]`（`engines`、`positive_species_marker`、`result_file`）与 `[engine.pfind]` / `[engine.diann]` / `[engine.alphadia]`、可选 `[entrapment]` 描述多引擎合并；其 `result_file` 输出的 JSON 即 `config.ini` 中 `search_engine_type=0` 可直接读取的输入。

## `constant/keys.py` 常量用途

- `ConstantsClass`（metaclass）：禁止改写属性（`__setattr__` 抛 `TypeError`），并提供 `get_values()` 返回类内全部用户定义的字符串值。
- `ConfigKeys`：集中所有访问 config 的字符串键，避免散落的魔法字符串。涵盖输入键（`INPUT`/`RAW_NUM`/`RAW_PATH`/`LIGHT_RESULT_PATH`/`SEARCH_ENGINE_TYPE`/`PFIND_QVALUE_THRESHOLD`）、通用键（`GENERAL`/`WORK_DIRECTORY`/`MASS_TOL_PPM`/`XIC_CYCLE_WINDOW`/`RESULT_FILE`/`FEATURE_TYPE`/`RANDOM_SEED`）与 mzML centroiding 键（`CENTROID_ENABLED`/`CENTROID_REL_THRESHOLD`，由 `manager/data_manager.py` 读取）。

## 边界 / 坑

- **缺 section 的 fallback**：`config.get("general", "work_directory", fallback="./workspace")` 中，`configparser` 的 `fallback` 只覆盖“section 存在但缺 option”；若整个 `[general]` 缺失会抛 `NoSectionError`，故 `main.py` 显式 `try/except NoSectionError` 再回退 `./workspace`。
- **独立工作目录**：允许每个 baseline 的 config 设独立 `work_directory`，避免并行 make 时多个 pipeline 写同一目录。
- **banner 容错**：`show_start_banner()` 读不到 `./banner/welcome.txt` 时回退打印内置 ANSI 彩色文本（不抛异常）。
- **日志父目录**：仅当 `--logpath` 含目录分量时才 `makedirs`，纯文件名（如 `ms2.log`）不触发。
