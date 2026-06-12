# ms2-met — 整个项目概览（L1）

> 全仓库 L1 概览。各子系统细节见 `docs/code/parts/<子系统>/{L2_role,L3_details,L4_api}.md`；
> 谱库读取模块有更细的独立文档，见 `docs/speclib/L1_overview.md`。

## 目标

蛋白质组学里，搜索引擎（DIA-NN/AlphaDIA/pFind 等）靠匹配理论谱图鉴定肽段，质量控制主流是 TDA（target-decoy）。**ms2-met 提供一种基于代谢标记（SILAC）的独立验证**：同一肽段有轻标/重标两种形式，二者 RT 相近、XIC 峰形相似。对搜索引擎给出的轻标鉴定，按序列算出重标 m/z，在 DIA 谱图里找重标证据——若轻/重标 XIC 峰形高度相关（Pearson 高）则鉴定可信，否则疑似假阳性。

**本工具对 DIA 数据下的搜索结果做特征提取**，输出轻重标 XIC 相关性等特征到 CSV，供后续训练/分析。

## 架构 / 数据流

```
config.ini → main.py → PairFlow(workflows)
   ├─ LightResultManager → LightResult（PSM 列表；DIA-NN/AlphaDIA/pFind/自定义JSON）
   └─ DataManager        → DIAData（mzML / PFB → npz，内存映射共享）
        ↓ 按 (sequence, charge, mods) 分组；生成正样本(label=1) + 负样本(label=0, RT+10min)
        ↓ 多进程特征提取（≤25 workers，single_work）
            对每个 PSM/PSM对：前体 MS1 XIC → 预测重标 m/z → 重标 MS1 XIC →
            前体 Pearson；逐 b/y 碎片：轻标 MS2 XIC vs 重标 MS2 XIC 的 Pearson；分位数统计
        ↓ 汇总 DataFrame → result.csv
（下游）tools/extract_common 构造多引擎交并集数据集；tools/spec_trainer 训练分类器打分
```

## 子系统索引（part = 子系统目录）

| 子系统 | 职责 | 文档 |
|---|---|---|
| entry | 程序入口 main.py + 常量/banner | [parts/entry](parts/entry/L2_role.md) |
| spectrum | 谱图数据核心：DIA 数据/XIC、PSM/重标质量、结果解析 | [parts/spectrum](parts/spectrum/L2_role.md) |
| workflows | 工作流编排 + 特征提取核心（PairFlow→single_work）| [parts/workflows](parts/workflows/L2_role.md) |
| manager | 数据管理层（Pickle 持久化基类 + 各 manager）| [parts/manager](parts/manager/L2_role.md) |
| tools | 顶层 CLI/工具（extract_common、eval_*、entrapment_classify）| [parts/tools](parts/tools/L2_role.md) |
| spec_trainer | 训练/评估分类器（多后端 lgb/xgb/sklearn）| [parts/spec_trainer](parts/spec_trainer/L2_role.md) |
| **speclib** | **pFind 谱库二进制读取（独立模块）+ 预测强度特征接入 feature_type=0（Phase 2a/2b/2c：I1/I2/I3/J2/J5）** | [docs/speclib/L1_overview.md](../speclib/L1_overview.md) |

## 关键技术 / 约定

- **多进程**特征提取（≤25 workers）；DIAData 经 npz 内存映射跨进程共享，峰用稀疏存储。
- **Pickle 持久化**：`manager/base_manager.py` 基类缓存解析结果（原始数据/搜索结果），避免重复解析。
- **SILAC 重标质量**：K +8.014204、R +10.008275（13C/15N）；亦支持 C-only/N-only 重标（`HeavyType`）。
- **质量容差** `mass_tol_ppm`、**XIC 窗口** `xic_cycle_window` 等由 config.ini 控制。
- **负样本**：同一肽段 RT 偏移（+10min）构造 label=0，模拟"找错位置"。
- 工作目录 `work_directory`（缺省 `./workspace`）可每个 baseline 独立，避免并行写冲突。

## 入口 / 快速上手

```bash
python main.py --configpath config.ini --logpath ms2.log
```
- 搜索引擎类型由 `config.ini` 的 `search_engine_type` 决定（0 自定义JSON / 1 DIA-NN / 2 AlphaDIA / 3 pFind）。
- 输出 `result.csv`（基本信息列 + XIC 相关性等特征列）。

## 文档体系约定

- **L1**（本文）= 整个项目；**L2** = 各子系统职责与接口；**L3** = 各子系统细节；**L4** = 逐"具体文件"API。
- 旧代码已全部纳入本体系（`docs/code/`）；`speclib` 模块单独维护一套更细的 per-module 文档（`docs/speclib/`）。
- 设计文档/实现计划另见 `docs/specs/`、`docs/superpowers/plans/`。
