# 可靠切割 R × 旧非空计数消融

本实验检验：在固定 neg20 拟合池、既有分组及训练协议下，轻重配对对最强共同采样点的依赖，是否提供现有 Q/D/S 之外的信息；移除三个旧非空计数输入是否影响这一增量。

设计源于此前对 109 条持续 FN 的谱图审阅，本页记录本次实现的完整特征定义和实验协议。本轮是开发验证：这些测试数据已被用于发现假设，不能将结果称为独立确认，也不能预先宣称新特征降低 FNR。

## 一键运行

在已有 neg20 原始输入、谱图和完整参考实验的机器上执行：

```bash
make 2da-reliability-ablation
```

依次执行参考完整性检查、全量 B/Q/D/S/R 提取、冻结实验构建、四臂训练与报告。即使使用 `make -j`，这些阶段也保持顺序。所有臂重新训练，共 5 个外层折 × 4 臂 × 5 个成员 = 100 个模型。

| 设置 | Make 变量 | 默认值 |
| --- | --- | --- |
| 原 neg20 提取配置 | `RELIABILITY_CONFIG` | `runs/baseline_2da_neg20/config.ini` |
| 冻结 neg20 参考实验 | `RELIABILITY_REFERENCE` | `runs/spec_trainer/single-peptide-structure-neg20` |
| 新特征目录 | `RELIABILITY_OUTPUT` | `runs/baseline_2da_neg20_reliability` |
| 新实验目录 | `RELIABILITY_ROOT` | `runs/spec_trainer/single-peptide-reliability-ablation` |
| 配对 bootstrap 次数 | `RELIABILITY_BOOTSTRAPS` | `1000` |

参考目录不是一张预测表：需要保留 `cohort.csv`、`neg20_pool.csv`、两个 manifest、`folds/`、`configs/`、`augmentation_audit.json`、`protocol.json` 和 `artifact_checksums.json` 等完整冻结文件。无需重跑旧实验；导入的旧配置可以保留服务器绝对路径，新实验会重写自己的路径。导入时校验旧输入文件，不要求历史实现代码与当前代码相同；新实验则冻结并校验当前实现代码。

使用上传结果作参考、在本机实际可读的原始数据上提取时，例如：

```bash
make 2da-reliability-ablation \
  RELIABILITY_REFERENCE=/home/verden/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/feature_result/ms2-met-runs-09-07/spec_trainer/single-peptide-structure-neg20 \
  RELIABILITY_CONFIG=/path/to/baseline_2da_neg20/config.ini
```

最后一个路径需要替换成运行机器上的配置；其中输入 JSON、PFB/谱库路径也必须可读，并与旧实验的原始输入快照一致。`neg20` 是错误鉴定筛选到 q≤0.20，不代表 trap 占样本总数的 20%。本实验仍使用真实 neg20 trap，没有新增人工负样本。

## 四个实验臂

| 臂 | 特征 | 用途 |
| --- | --- | --- |
| `a0` | B+QDS | 同环境重新训练的基线 |
| `a1` | B+QDS+R | R 的增量，主比较 a1−a0 |
| `a2` | B+QDS，去除 `b_count/y_count/all_count` | 指定三个旧计数的直接作用 |
| `a3` | a2+R | 去除旧计数后的 R 增量 |

移除特征发生在重新训练之前，不能把已有模型输入置零代替消融。保留 Q 的逐电荷配对计数及其它特征，因此 a2 不声称消除了所有计数代理。

四臂使用同一 q01 基础样本、相同 neg20 增强池及合法成员拟合行。外层约 80% 的 group 用于开发、20% 留作该折测试；内部另有拟合、早停和 OOF 校准隔离，不能把外层 80% 全部称为某个成员的拟合行。各折轮换后，每个 q01 样本恰好得到一次外层测试预测。

沿用现有 peptide/生成关系连通 group 与 I/L 规范化，不按 rep 重分。复制外层 manifest，保留内部 OOF 折号和早停 mask。新增 neg20 仅用于拟合，继续排除与该成员早停、OOF 校准及外层测试 group 相连的行。超参数、类别权重、AUC 早停目标和校准规则保持参考配置不变。

## R 的定义

实现于 [fragment_reliability.py](../../workflows/fragment_reliability.py)，由现有 [fragment_structure.py](../../workflows/fragment_structure.py) 在原始主组确定后调用。只用单个候选自己的序列、轻重逐电荷 XIC 和峰身份，不使用竞争候选或蛋白来源。

对原 QDS 主组内的每个独立轨迹组：

1. 继承原 separable 判断、逐电荷配对、实际 scan/centroid 精确去重及 ordinal 处理。
2. 在轻重共同的真实采样 cycle 上，找归一化轻重强度乘积最大的 cycle；并列取最早。
3. 从两侧同时去掉这一 cycle，仅作特征计算；保留其它 cycle 编号及 RT，不插值、不重编号。
4. 重用生产配对规则，检查剩余轻重强度、apex 间隔和 Pearson。不能仅凭 Pearson 决定存活。
5. 将仍通过配对的无歧义切割加入集合。一个切割有多条独立证据时，任意一条存活即可；同一切割计数一次。一个峰组对应多个不同切割位置时，不给任一位置确定支持。

不重新选择主组。长度为 N、保留切割集合为 C 时：

```text
ms2_reliability_stable_cut_fraction = |C| / (N - 1)
ms2_reliability_stable_longest_gap_fraction
    = max(diff(sorted({0, N} ∪ C))) / N
```

完整采集但主组为空时是 0 覆盖、1 缺口。未采集、无可分离目标、无实际峰身份、跨 run 等情况是 NA，并输出 `fragment_reliability_valid/version/status`，版本为 `r_v1`。状态字段不进入正式模型。NA 行继续保留，不能以 R 好坏筛掉测试行。

R 是支持稳定性的描述，不是错误真值。真实窄峰也可能依赖单个采样点；结果必须同时评估正确鉴定被误拒绝的损伤。

## 输入一致性与续跑

原 Q/D/S 的定义、版本及 B 特征不变。普通旧入口默认不输出 R；新提取入口显式开启 `fragment_reliability_features=true`，使用相同逐离子面板，不为 R 重读扫描。

构建器先在完整输入关系图上复核增强来源和桥接排除，要求新增提取所得 neg20 池与参考完全一致。原 q01 及增强池的样本 ID、label、q_value、B/QDS 数值、结构状态和关系字段必须一致；浮点比较容差 rtol=atol=1e-9。R 还检查版本、缺失状态、范围、切割离散性和最长缺口可行性：R 不能增加原主组切割或缩短其最长缺口。

构建时逐成员核对拟合行审计及哈希。训练完成后再次核对实际增强、特征列、OOF 行、模型路径、阈值和多数票。输出新目录，不覆盖旧实验。

```bash
make 2da-reliability-ablation-features   # 只提取全量 B/QDS/R
make 2da-reliability-ablation-build      # 只构建冻结实验；拒绝覆盖
make 2da-reliability-ablation-train      # 训练或续跑冻结实验
make 2da-reliability-ablation-summarize  # 校验并汇总
make 2da-reliability-ablation-verify     # 校验冻结输入与当前实现
```

已有新 R 特征时可跳过提取，直接构建、训练并汇总：

```bash
make 2da-reliability-ablation-run \
  RELIABILITY_FEATURES=/path/to/reliability/features.csv \
  RELIABILITY_REFERENCE=/path/to/single-peptide-structure-neg20
```

一键入口再次运行时，仅复用输入、配置、代码及特征 SHA-256 全部一致的完成提取；完成训练任务经验证后跳过，未完成任务重跑。原始大体积 PFB/谱库按不可变输入管理，不在每次续跑时全量哈希。更改这些文件时要换输入快照和输出目录。

旧的 QDS-only 输出不能冒充 R 输出。提取中断并留下不完整 CSV，或实现/输入发生变化时，应使用新提取及实验目录。仅改 `RELIABILITY_FEATURES` 适用于 `-run/-build`；一键入口实际提取目的地由 `RELIABILITY_OUTPUT` 决定，两者须指向同一结果。

## 评价与判断

保持 label=1 正确、label=0 错误及模型 trust 输出。评价显式转换 `error_truth=1-label`、`error_score=1-trust`，错误鉴定为统计正类，复用 `cv_core.py`。

成员在自己的合法 OOF 正确鉴定上校准 FPR1/5/10，外层测试按成员多数票决定是否标错。报告实际测试 FPR、FNR、混淆矩阵、错误召回、`roc_auc`、`error_pr_auc`、`fnr_at_fpr5`、`error_recall_at_fpr10`。名称中的 FPR5 是训练校准目标，并不保证实际测试 FPR 恰好 5%；不能把成员 OOF 阈值直接套在平均 trust 上。

预声明比较为 `reliability_full`（a1−a0，主比较）、`drop_counts`（a2−a0）、`reliability_pruned`（a3−a2）、`combined`（a3−a0）及 `interaction`（a3−a2−a1+a0）。同一轮 bootstrap 以冻结 group 为单位、按真实类别分层，为所有臂使用相同重采样权重，报告错误召回及实际 FPR 差异。预测固定，不重训；区间是开发阶段的描述，次比较没有多重校正，模型间标准差不充当置信区间。

主比较沿用参考开发门槛（当前参考：错误召回增加至少 3 个百分点、召回差区间下界>0、实际 FPR 差区间上界≤+0.5 个百分点）。评估全体冻结正确样本和 trap，记录找回 FN、新增 FN、找回 FP、新增 FP；不只统计原 109 条 FN 的恢复数。相同分组防止样本泄漏，但不能消除先看过测试集再提出假设的适应，最终确认仍需要未用于特征设计、排除 peptide/group 重合的新数据。

## 输出

- `report.md`、`summary.csv/json`：四臂指标、主比较判断与解释。
- `paired_comparisons.csv`、`transitions.csv`：五个配对比较、区间、逐样本 FN/FP 转移。
- `pooled_test_predictions.csv`：每个 q01 样本的四臂外层测试结果。
- `reliability_audit.json`、`augmentation_audit.json`：R 可用性、原列保持情况与逐成员拟合池审计。
- `cohort.csv`、`neg20_pool.csv`、`folds/`、`configs/`、`training/`：冻结输入、配置和模型。
- `protocol.json`、`artifact_checksums.json`、`bundle_status.json`：协议、校验与运行状态。

实验实现：[fragment_reliability_ablation.py](../../tools/fragment_reliability_ablation.py)。相关反例与流程测试：[特征测试](../../tests/test_fragment_reliability.py)、[四臂集成测试](../../tests/test_fragment_reliability_ablation.py)。

## 验证记录

2026-09-19：140 项相关测试通过，覆盖 QDS/R 提取、特征注册、旧 QDS 与 neg20 流程、四臂隔离、数据漂移拒绝、缺失值、FP/FN/FPR/FNR 数值口径和续跑。唯一警告来自既有 CV 测试刻意设置的某类样本少于折数。测试命令：

```bash
python -m pytest \
  tests/test_fragment_structure.py tests/test_fragment_reliability.py \
  tests/test_fragment_structure_validation.py tests/test_fragment_structure_neg20.py \
  tests/test_fragment_reliability_ablation.py tests/test_cv_core.py \
  tests/test_feature_groups.py tests/test_feature_cols_contract.py -q
```

另通过实际 `make -j4 2da-reliability-ablation-run`，使用 200 条 q01 和 70 条额外候选的构造夹具完成 100 个 LightGBM 模型，再次运行全部模型哈希不变。这是训练 CLI 与报告流程测试；提取环节另行核对，不能把夹具模型的分数当作真实效果。测试环境使用临时隔离安装的 LightGBM 4.6.0。

真实数据核查：

- 上传的 neg20 参考目录通过 42 个冻结文件 SHA-256 校验；107,227 条 q01 与 5,538 条 neg20 增强行的原特征、标签和关系字段通过新构建器的一致性检查。
- 从原始 PFB 重新读取此前审阅的 260 条记录，共核查 16,120 个 MS2 扫描记录（包含候选之间重复读取）。7,280 个原 Q/D/S 值与原结果一致；两个生产版 R 特征与先前原型逐条一致，最大绝对差为 0。
- 单候选流水线测试确认开启 R 后复用同一逐离子面板，原特征不变、不增加谱图扫描读取次数。

本机默认 `runs/baseline_2da_neg20/config.ini` 和 `datasets/hela_2da_neg20.json` 当前不存在，因此本轮没有启动真实全量提取或四臂训练。需要在输入齐全的运行机器上执行一键命令，或指定与参考快照兼容的实际配置。上述一致性检查证明实现符合定义，不证明 R 能降低真实 FNR。
