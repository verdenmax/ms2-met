# ms2-met

> 利用代谢标记（SILAC、全 ¹³C、全 ¹⁵N）发展的 DIA MS2 检验技术 —— 对搜索引擎的鉴定结果做**正交独立验证**与特征提取。
>
> Independent, orthogonal DIA MS2 validation and feature extraction powered by metabolic labeling (SILAC, uniform ¹³C, or uniform ¹⁵N).

📖 **可视化使用说明 / Visual usage guide:** open [`docs/usage.html`](docs/usage.html)
（顶部可切换 **A 终端风 / C 科研风** 两种风格 · toggle between Terminal / Science themes）

---

## 这是什么 / What is it

搜索引擎（DIA-NN / AlphaDIA / pFind）靠匹配理论谱图鉴定肽段，质控主流是 target-decoy (TDA)。
**ms2-met 提供一条正交验证**：同一肽段的轻标 / 重标两种形式 RT 相近、XIC 峰形相似。对一条轻标鉴定，
按配置的标记化学计算**重标 m/z**，在 DIA 谱图里找重标证据 —— 若轻 / 重标 XIC 峰形高度相关
（Pearson 高）则鉴定可信，否则疑似假阳性。最终把相关性等特征输出到 CSV，供训练 / 分析。

A search-engine identification of a *light* peptide implies a chemistry-specific *heavy* twin.
ms2-met computes its precursor/fragment m/z values, extracts XICs from the DIA run, and scores the light-vs-heavy
peak-shape correlation. High correlation ⇒ trustworthy ID; low ⇒ likely false positive. Features are
written to a CSV for downstream training / analysis.

Uniform ¹³C/¹⁵N mode currently supports unmodified peptides. PTM elemental
composition and a purity-aware uniform-label isotope-envelope model are not
implemented; unsupported modified PSMs fail explicitly, and the isotope
correlation feature is marked invalid instead of emitting a fabricated value.

---

## 数据流 / Pipeline

```
mzML (DIA raw) + 搜索结果(DIA-NN/AlphaDIA/pFind/JSON)
        │
        ▼   main.py  ──reads──▶ config.ini ──▶ PairFlow
        │      ├─ LightResultManager ─▶ LightResult (PSM 列表 / list)
        │      └─ DataManager        ─▶ DIAData    (mzML→npz, mmap 共享 / shared)
        ▼
   分组(seq,charge,mods) + 正/负样本(label 1/0)
        ▼
   多进程特征提取 (≤25 workers, single_work)
     前体 MS1 XIC → 预测重标 m/z → 重标 MS1 XIC → Pearson
     逐 b/y 碎片: 轻标 MS2 XIC vs 重标 MS2 XIC → Pearson + 分位数统计
        ▼
   result.csv  ──▶ tools/extract_common (多引擎数据集) ──▶ tools/spec_trainer (训练打分)

[独立模块 / standalone, 未接入 pipeline]
   pFind 谱库 (pepdata.pdb + rt/ms2.predb + FASTA + modification.ini)
        └─▶ spectrum/speclib (流式解码) ─▶ tools/speclib_inspect (质量自校验 CLI)
```

完整可视化版本见 [`docs/usage.html`](docs/usage.html)。Full interactive diagram in the usage guide.

---

## 快速上手 / Quickstart

```bash
# 1 · 主流程特征提取 / run feature extraction → result.csv
python main.py --configpath config.ini --logpath ms2.log

# 2 · 批量跑多窗口 baseline / Makefile targets
make 2th        # 2Da 窗口
make 5th        # 5Da 窗口
make normal     # 变窗 / variable window
make all        # 顺序全跑 / run all

# 3 · 构造多引擎交并集数据集 / build union dataset
python tools/extract_common.py --configpath extract_2da_pfind_diann.ini

# 4 · 训练 / 评估分类器 / train scorer
cd tools/spec_trainer
python src/main.py --config config/<name>.yaml --name <name>

# 5 · 读取 pFind 谱库（独立工具）/ read spectral library (standalone)
python -m tools.speclib_inspect \
    --library-dir <谱库目录> \
    --fasta merge_human_ecoli_yeast.fasta --mod modification.ini \
    --element element.ini --aa aa.ini --mass-limit 300000

# 6 · 构造训练专用 synthetic hard negatives（两阶段，中间运行外部 DIA 搜索）
cp training_set_builder.ini.example training_set_builder.ini
python -m tools.training_set_builder generate --config training_set_builder.ini
# search generated FASTA in TRAIN raws, then extract ordinary features.csv
python -m tools.training_set_builder assemble --config training_set_builder.ini
```

正式的 MS1/MS2 消融使用同一 eligibility 共同队列、按 `sequence`
分组的 5 折 CV，以及注册表中的固定特征组。先在 2da 上预跑，再运行三种
采集条件的完整 neg20 矩阵：

```bash
cd /path/to/ms2-met

make train-ablation-neg20-2da \
  FEATURE_ROOT=/path/to/ms2-met-runs-08-08

make train-ablation-neg20 \
  FEATURE_ROOT=/path/to/ms2-met-runs-08-08
```

`FEATURE_ROOT` 必须直接包含 `baseline_2da_neg20/`、
`baseline_5da_neg20/` 和 `baseline_normal_neg20/`。外部特征文件只读；生成的
配置、模型和 JSON 结果位于 `runs/spec_trainer/ablation/neg20/`。历史
`train-cv-neg20-all` 不执行正式 feature-arm 消融。

依赖安装 / install deps: `pip install -r requirements.txt`（本机推荐用 conda 管理环境）。

---

## 关键配置 / Key config (`config.ini`)

| 键 / key | 说明 / description |
|---|---|
| `search_engine_type` | `0` 自定义 JSON / `1` DIA-NN / `2` AlphaDIA / `3` pFind |
| `feature_type` | `0` 同文件配对（陷阱库负例）/ `1` 轻重标双文件 |
| `labeling` | `silac`（默认）、`c13`/`13c`/`cheavy`、`n15`/`15n`/`nheavy` |
| `mass_tol_ppm` | 质量容差（ppm），缺省 `10` |
| `xic_cycle_window` | XIC 提取的周期半窗 / XIC half-window in cycles |
| `centroid_enabled` / `centroid_rel_threshold` | 加载 mzML 时是否质心化 + 阈值（推荐 `1e-3`） |
| `work_directory` | 工作目录（缺省 `./workspace`）；每个 baseline 可独立以避免并行写冲突 |
| `result_file` | 输出 CSV 路径 / output CSV path |

---

## 组件速查 / Components

| 子系统 / subsystem | 职责 / role | 文档 / docs |
|---|---|---|
| `entry` | 程序入口 `main.py` + 常量 / banner | [L2](docs/code/parts/entry/L2_role.md) |
| `spectrum` | DIA 数据 / XIC、PSM / 重标质量、结果解析 | [L2](docs/code/parts/spectrum/L2_role.md) |
| `workflows` | `PairFlow → single_work` 特征提取核心 | [L2](docs/code/parts/workflows/L2_role.md) |
| `manager` | 数据管理层（Pickle 持久化缓存） | [L2](docs/code/parts/manager/L2_role.md) |
| `tools` | `extract_common` / `eval_*` / `entrapment_classify` / `speclib_inspect` | [L2](docs/code/parts/tools/L2_role.md) |
| `spec_trainer` | 训练 / 评估分类器（lgb / xgb / sklearn） | [L2](docs/code/parts/spec_trainer/L2_role.md) |
| `speclib` | pFind 谱库二进制读取（独立模块） | [L1](docs/speclib/L1_overview.md) |

---

## 文档体系 / Documentation

分层文档（L1 全局 → L4 逐文件 API）/ layered docs (L1 overview → L4 per-file API):

- [`docs/code/L1_overview.md`](docs/code/L1_overview.md) — 全项目概览 / whole-project overview；
  各子系统细化见 `docs/code/parts/<子系统>/{L2_role, L3_details, L4_api}.md`。
- [`docs/speclib/L1_overview.md`](docs/speclib/L1_overview.md) — 谱库 reader 独立 L1-L4 文档。
- `docs/specs/` · `docs/superpowers/plans/` — 设计文档与实现计划 / design specs & implementation plans。

---

## 测试 / Tests

```bash
python -m pytest tests/ -q
```
