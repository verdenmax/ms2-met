# neg20 拟合增强与 Q/D/S 配对验证

本实验在原 q01 正确鉴定和 trap 的冻结分组上，仅向拟合部分加入 `0.01 < q_value <= 0.20` 的真实 trap，比较增加错误训练样本及 Q/D/S 的效果。`neg20` 是 trap 搜索结果的筛选上限，不表示训练中 trap 占 20%。

后续可靠切割 R 与旧计数消融使用本实验的完整冻结目录作参考，单独运行 `make 2da-reliability-ablation`，见 [R 消融文档](2026-09-19-fragment-reliability-ablation.md)。本页原入口和四臂含义不变。

## 一键运行

在已有 neg20 输入 JSON、谱图和之前的结构验证目录的机器上执行：

```bash
make 2da-structure-neg20
```

先核查参考实验完整性，再依次完成 neg20 全量 Q/D/S 提取、输入一致性及分组审计、四臂训练、固定 q01 测试、配对比较。四臂均重新训练，共 5 个外层折 × 4 臂 × 5 个成员 = 100 个模型。新结果独立保存，旧实验不被覆盖。

| 设置 | 默认值 |
| --- | --- |
| neg20 原始提取配置 | `runs/baseline_2da_neg20/config.ini` |
| neg20 Q/D/S 特征输出 | `runs/baseline_2da_neg20_structure/features.csv` |
| 冻结参考实验 | `runs/spec_trainer/single-peptide-structure-validation` |
| 新训练及报告 | `runs/spec_trainer/single-peptide-structure-neg20` |

参考目录必须保留完整的 `cohort.csv`、`folds/`、`configs/`、两个 manifest、protocol、status 和 checksum 文件。其模型文件和报告不作为新训练输入；复制回本机的服务器结果允许保留原来的绝对路径记录，构建器会使用本地冻结文件，并重写新任务的数据与输出路径。

输入位于其他目录时：

```bash
make 2da-structure-neg20 \
  NEG20_STRUCTURE_CONFIG=/path/to/baseline_2da_neg20/config.ini \
  NEG20_STRUCTURE_REFERENCE=/path/to/single-peptide-structure-validation \
  NEG20_STRUCTURE_OUTPUT=runs/baseline_2da_neg20_structure \
  NEG20_STRUCTURE_ROOT=runs/spec_trainer/single-peptide-structure-neg20
```

提取配置中的 `input.light_result_file` 必须指向既有 neg20 JSON，PFB/谱库路径也必须在运行机器上可读。全量提取沿用该配置；不会把先前人工审阅的 130 个 FN 单独拿来提取或训练。若尚未生成默认 neg20 JSON，可先运行已有 `make extract-2th-neg20`，其配置是 `extract_2da_neg20.ini`。

`FEATURE_ROOT` 控制默认提取配置的父目录，`CV_OUTPUT_ROOT` 控制默认参考/结果父目录，也可直接使用上述具体变量。

## 分步与续跑

```bash
make 2da-structure-neg20-features    # 提取；已完成时校验后复用
make 2da-structure-neg20-build       # 只构建，不覆盖已有实验目录
make 2da-structure-neg20-train       # 训练或续跑
make 2da-structure-neg20-summarize   # 复核完整结果并汇总
make 2da-structure-neg20-verify      # 校验冻结文件及代码指纹
```

如果全量 Q/D/S 已经提取完，可以直接构建并训练：

```bash
make 2da-structure-neg20-run \
  NEG20_STRUCTURE_FEATURES=/path/to/neg20_structure/features.csv \
  NEG20_STRUCTURE_REFERENCE=/path/to/single-peptide-structure-validation
```

一键入口即使使用 `make -j` 也先完成提取，再启动训练。训练目录已有时先核查输入指纹、代码及 bootstrap 设置，校验通过的完整任务跳过；未完成任务重新执行。训练锁防止同一实验同时写入。

新提取结果记录输入 JSON、源配置、实际配置、实现代码及特征文件的 SHA-256，`--resume` 仅复用完整且指纹一致的结果。旧提取结果若没有这些指纹，应选择新的提取目录。原始 PFB 和谱库按不可变输入管理；它们变化时应使用新输入快照和输出目录。提取中断后若已产生不完整 features.csv，程序会拒绝把它当作完整结果；确认失败原因后使用新提取目录重跑。

## 四个训练臂

| 臂 | 基础拟合样本 | 新增拟合样本 | 特征 |
| --- | --- | --- | --- |
| `b` | 原 q01 | 无 | 当前 B |
| `b_qds` | 原 q01 | 无 | B+QDS |
| `b_neg20` | 原 q01 | 合格的新增 neg20 trap | B |
| `b_qds_neg20` | 原 q01 | 同一份新增 neg20 trap | B+QDS |

所有臂逐字节复制原外层 train/test CSV，保留其内部 OOF 折号和各成员早停 mask。模型超参数、训练目标、early stopping 配置使用原参考任务配置。首轮不修改类别权重、AUC 早停指标或阈值规则。

四臂使用同一运行环境重新训练，FN 转移以本轮各比较的基线为准。若 LightGBM 等运行环境与历史实验不同，不应自动假定新 B 的 FN 集合仍恰好是此前审阅的 130 条。

新增样本经过原来的 `evidence_observed` 筛选，不根据 Q/D/S 好坏再筛选个体。数学上不适用的字段允许 NA，但必须有完整的提取状态和版本；未提取 Q/D/S 的旧 CSV 不能填全 NA 后混入训练。若增强池完全没有可用 Q/D/S，会拒绝运行。

重提取结果必须包含全部原冻结样本；这些样本的 B、Q/D/S、q_value、label 和结构状态必须与参考一致。数值容差为 rtol=atol=1e-9。新增正确样本不加入模型，q01 之外的 trap 也不进入正式测试。

## group 隔离

构建器把新增输入和原 `source_outer_fold_manifest.csv` 的全量关系图连接起来。原非训练候选、父样本和 orphan 等关系桥仍参与图审计；新增输入在 cohort 过滤之前参与连接检查。

- 序列使用现有 I/L 规范化，保留 peptide、parent、candidate-family 等关联。
- 新输入若把多个旧冻结 group 连接起来，该连通分量中的新增候选全部排除并记录，不改变旧 group 或测试集。
- 与一个旧 group 相连的新增 trap 继承旧 group，外层测试到该 group 时不加入拟合。
- 每个成员再排除与其早停或 OOF 校准 group 相连的新增 trap。
- 完全不与原 q01 组相连的新增 group 可进入所有成员的拟合部分；它们不属于固定的 q01 评估集合，也不用于早停或校准。

CV 训练器支持 `data.fit_augmentation_files`，目前仅接受 `negative_source=real_entrapment_neg20` 且 label=0、0.01<q≤0.20 的行。在任何模型拟合前检查外层 group 隔离和可见关系的一致性，实际拟合时按成员再次排除早停/校准组。

原 OOF dataframe 不追加增强行，OOF 样本数和顺序保持不变。每个成员保存实际增强行数、group 数、排除数量、样本 ID 集合哈希及最终拟合比例；完成任务复核时必须与构建期记录一致。

## 指标与解释

存储保持 label=1 正确、label=0 错误；模型输出 trust。评价显式使用 `error_truth=1-label` 和 `error_score=1-trust`，错误鉴定为统计正类。复用 `cv_core.py` 的阈值及混淆矩阵函数。

每个成员仍在原真实 OOF 的正确样本上校准 FPR1/5/10，然后在原 q01 外层测试上按成员多数投票。FPR5 名称表示训练侧目标，报告必须同时展示实际外层 FPR 和 FNR。均值分数的 ROC-AUC 和 error PR-AUC 仅描述排序，不替代正式投票工作点。

五个预声明比较：

| 比较 | 计算 | 解释 |
| --- | --- | --- |
| `augmentation_base` | b_neg20 − b | 增加真实 trap 对 B 的收益 |
| `augmentation_qds` | b_qds_neg20 − b_qds | 增加真实 trap 对 QDS 模型的收益 |
| `structure_q01` | b_qds − b | 原条件下 Q/D/S 增量 |
| `structure_neg20` | b_qds_neg20 − b_neg20 | **主比较：增强后 Q/D/S 增量** |
| `interaction` | structure_neg20 − structure_q01 | 增强是否改变 Q/D/S 的增量 |

对错误召回与实际 FPR 分别计算上述差异。bootstrap 按原冻结 group、按真实类别分层，同一轮重采样权重应用于全部模型和比较；原预测固定，不重新拟合。区间不是折间标准差，探索性比较区间没有多重校正。主比较沿用参考实验的实际增益与 FPR 容许上限，另要求召回差区间下界大于 0、FPR 差区间上界不超过容许值。默认 bootstrap 1,000 次，可通过 `NEG20_STRUCTURE_BOOTSTRAPS` 在构建前设置。

`transitions.csv` 同时记录找回 FN、新增 FN、新增 FP 和找回 FP。interaction 直接做差并重采样，不能根据“一组显著、另一组不显著”推断交互。

新增 neg20 改变了训练错误的数量和类型，因此效果不能单独归因为类别比例。该 q01 数据已用于审阅和特征开发，本轮仍是开发验证；构造负样本增强尚不属于本轮输入。

## 结果文件

- `report.md`、`summary.csv/json`：四臂排序与正式 FPR/FNR、比较及解释。
- `paired_comparisons.csv`：五个配对比较、区间、FN/FP 转移数量。
- `pooled_test_predictions.csv`、`transitions.csv`：同一冻结测试集上的分数、投票与转移。
- `augmentation_audit.json`、`neg20_pool.csv`、`rejected_bridges.csv`：全局及逐成员增强审计。
- `folds/`、`configs/`、`training/`、protocol/checksums/status：冻结输入、任务配置、模型和完成状态。

实现：[fragment_structure_neg20.py](../../tools/fragment_structure_neg20.py)、[CV 训练器](../../tools/spec_trainer/src/cv_train.py)。输入预检记录见 [augmentation_experiment_plan.md](../../analysis/fragment_structure_validation_2026_09_13/augmentation_experiment_plan.md)。

## 验证记录

2026-09-13：112 项相关测试通过，覆盖原 CV/QDS 流程、指标口径、负样本来源约束、数据漂移拒绝、全图桥接、I/L 等价分组、仅拟合增强、配对交互比较及续跑。另用 200 行 q01 测试夹具和 70 行新增 trap 实际运行 `make -j4 2da-structure-neg20`，完成四臂的 100 个 LightGBM 模型；再次运行跳过 20 个已完成任务，全部模型 SHA-256 不变，汇总 JSON 可严格解析。这些构造数据仅验证软件流程；本机测试使用 LightGBM 4.7.0。

上传的真实参考实验通过全部 40 个冻结文件校验。本机默认提取配置所需的 `datasets/hela_2da_neg20.json` 和对应默认 DIA-NN parquet 不存在，因此没有启动真实全量提取/训练。应在输入齐全的运行机器上执行上述命令，或先补齐与参考快照兼容的输入，再通过全量特征一致性检查。
