# workflows — 职责与接口

## 一句话职责

SILAC 轻重标 MS2 校验工具的**编排与特征提取层**：从配置/搜索结果出发，把每条 PSM 的轻重标在 DIA 谱图中配对，多进程抽取一整套 XIC 特征，最终落盘为训练用 `features.csv`。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `PairFlow` | `class(workname, config, work_path)` | 顶层工作流；`main.py` 唯一入口，`run()` 跑全流程 |
| `PairFlow.run()` | → `None` | `load()` 读数据 → `distribute()` 多进程算特征写 CSV |
| `PairFlow.load()` | → `None` | 加载 light result（DIA-NN 等）与 `DataManager` |
| `PairFlow.distribute()` | → `None` | 按 `feature_type` 分发任务到进程池，汇总写 `result_file` |
| `multi_batch_work(psm1, dia_data1, psm2, dia_data2, config)` | → `dict` | **双文件/双 PSM** 特征提取（跨 run 轻重标） |
| `single_pair_work(psm, dia_data, config)` | → `dict` | **单文件** 特征提取（同一 raw 内推出 heavy） |
| `calc_xic_score(light_xic, heavy_xic, ...)` | → `dict` | 一对轻重标 XIC 的核心打分（19 字段） |
| `data_to_npz(mgr, filepath, workpath)` | → `(name, shared_path)` | 把 DIA 数据缓存成 `.dia.npz`（mmap 共享） |
| `process_batch_single / _pair / _pair_shuffle(...)` | → `(results, n_errors)` | 进程池批处理工作函数（三种 `feature_type`） |
| `Q1aAccumulator` | `class(split_window, ...)` | 逐 PSM 累计 b/y 碎片配对召回，输出 11 个 q1a 特征 |
| `pred_features.*`（谱库预测强度，Phase 1） | 纯函数 | `spectral_angle`/`spearman_sim`/`select_topk_separable`/`i1_pattern_features`（轻重关系特征底座） |
| `pred_store.*`（谱库预测强度，Phase 1） | 肽段→预测 lookup | `normalize_key`/`frag_key`/`build_pred_store` → `PredStore`（一遍流式、O(命中)） |

## 依赖

- 依赖：`spectrum`（`DIAData`、`PSMInfo`、SILAC 质量/同位素工具）、`manager`（`DataManager`、`LightResultManager`）、`constant.keys.ConfigKeys`；数值栈 `numpy`/`scipy`，进度条 `rich`。
- 被依赖：`main.py` 直接构造并运行 `PairFlow`。

> **谱库预测强度特征（Phase 1 基础，未接入主流程）**：`pred_features.py`（度量/top-K/I1 纯函数）、`pred_store.py`（肽段→预测一遍流式 lookup）与 `tools/speclib_sanity.py`（前置 go/no-go gate）已就绪，仅供 CLI 与单测；接入 `result.csv` 见 `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md` 的 Phase 2。

## 输入 / 输出

- 输入：`config.ini`（`[input]` raw 路径数 `raw_num`、`raw_path_N`、`light_result_file`；`[general]` `mass_tol_ppm`、`xic_cycle_window`、`feature_type`、`result_file`、`random_seed`、`work_directory`），DIA raw 文件，搜索结果文件。
- 输出：`result_file`（CSV，每行 = 一条 PSM/PSM 对的元数据 + 全部特征 + `label`）；进程池崩溃时旁路写 `*.PARTIAL_INCOMPLETE` 标记；工作目录下的 `*.dia.npz` 缓存。
