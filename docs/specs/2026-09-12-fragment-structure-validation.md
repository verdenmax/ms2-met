# Q/D/S 冻结分组五臂验证流水线

状态：已实现构建、训练/续跑、核查和汇总入口，并完成小型正式训练器端到端测试。尚未运行真实数据的全量 Q/D/S 提取或五臂训练，没有新的真实 FPR/FNR 结论。

## 提取之后的命令

```bash
make 2da-structure-features
make 2da-structure-validation
```

第二个命令依次完成：读取原冻结样本与分组 → 按样本身份合并新特征 → 冻结内部划分 → 五个特征组训练 → 逐行配对比较与报告。它不重新提取谱图，不重新生成负样本，不重新划分外层 group，也不因 Q/D/S 缺失而删样本。

默认需要这些输入：

| 输入 | 默认路径 | 用途 |
| --- | --- | --- |
| 新特征 | `runs/baseline_2da_structure/features.csv` | 提供新提取的 28 个 Q/D/S 字段 |
| 原特征快照 | `runs/baseline_2da_clean/features.csv` | 对应原 manifest 的 source_row，验证样本身份及原观察特征没有漂移 |
| 原冻结 manifest | `runs/spec_trainer/counterfactual-2da-real-q01-effectiveness/outer_fold_manifest.csv` | 继承已有样本 ID、connected group 和五个外层折 |
| 验证配置 | `config/fragment_structure_validation.yaml` | 训练模板、bootstrap 和预声明比较条件 |

这里保留两个特征 CSV 是为了将新信息接到原冻结实验上：新 CSV 可重排，原快照必须与原 manifest 对应。不能把另一份同行数、同序列但不同 PSM 的旧文件替换进去。

若历史产物在其他目录，显式指定：

```bash
make 2da-structure-validation \
  STRUCTURE_VALIDATION_FEATURES=runs/baseline_2da_structure/features.csv \
  STRUCTURE_VALIDATION_SOURCE_FEATURES=/path/to/original/baseline_2da_clean/features.csv \
  STRUCTURE_VALIDATION_MANIFEST=/path/to/original/outer_fold_manifest.csv \
  STRUCTURE_VALIDATION_ROOT=runs/spec_trainer/single-peptide-structure-validation
```

`STRUCTURE_VALIDATION_CONFIG` 可指定另一份实验配置；`PY` 沿用原 Makefile 的 Python 环境设置，需要已有训练环境中的 LightGBM。配置中的 `training_template` 相对于该 YAML 所在目录解析。

## 单步执行与续跑

```bash
make 2da-structure-validation-build       # 只核查输入、合并和冻结配置
make 2da-structure-validation-verify      # 核对冻结文件与代码指纹
make 2da-structure-validation-train       # 训练或续跑
make 2da-structure-validation-summarize   # 汇总完整结果
```

`build` 拒绝覆盖已有输出目录。一键 `2da-structure-validation` 在目录已存在时核查输入、模板和代码指纹，然后续跑；输入、特征定义或训练实现变化时需选择新输出目录。

续跑单位是“一个外层折 × 一个特征组”，共 25 个任务。每个任务由正式训练器训练 5 个成员，因此完整实验共 125 个成员模型。已完成且核查通过的任务跳过；未完成任务可重跑，其内部 5 个成员会一起重新训练。完整结果损坏时直接报错，不默默覆盖。

训练锁 `.train.lock` 防止两个进程同时训练同一输出目录。普通异常会清理锁；机器宕机或强制终止留下锁时，应先确认记录的 PID 已不在运行，再清理该残留锁。冻结 bundle 中路径为创建时的绝对路径；移到另一机器后需要重新 build 训练输入，已生成的汇总表可直接阅读。

## 样本、分组和五个特征组

只使用 manifest 中 `experiment_origin=real_q01` 的原冻结行。当前已核查数据为 107,227 条、106,666 条标注正确、561 条 trap、38,192 个原 group；代码不硬编码这些数量，实际输出数量会写入审计。

原 CSV 的行号只用于回到它所对应的原快照，然后重建原 `experiment_sample_id` 并核查序列、标签。新 CSV 完全通过该身份匹配，不使用新行号。新输出中的额外行不进入冻结集合，但记录数量；漏行、身份重复、分组跨折、源快照不匹配、原观察特征变化均拒绝构建。

原观察特征以原快照为基准，并逐字段检查新提取值相等（允许 CSV 浮点往返误差：rtol/atol 均为 1e-9）。原 `evidence_observed` 可计算性规则必须对冻结集合仍全部成立。新 Q/D/S 不可用时保留 NaN 和状态，不改变 cohort。全部 Q/D/S 不可用、版本混用、非法数值或状态与缺失值矛盾时拒绝开始一个无效实验。

| 臂 | 注册特征组 | 当前列数 |
| --- | --- | ---: |
| B | `ms1_ms2_no_prediction` | 133 |
| B+Q | `ms1_ms2_charge` | 146 |
| B+D | `ms1_ms2_dedup` | 136 |
| B+S | `ms1_ms2_structure` | 145 |
| B+QDS | `ms1_ms2_qds` | 161 |

可计算性标志、状态、标签和分组字段不作为模型输入。`require_complete_arm=true`，列清单和顺序冻结在 `protocol.json` 中，每个训练结果都核对实际所用列。

五个外层折完全继承 `leakage_group_id` / `experiment_outer_fold`，包含已有家族关系，不按 Rep 留出。每个外层折内部只为真实训练数据生成一次五成员 OOF 分配和独立早停组，并让五个特征组共用。拟合、早停、OOF 校准、外层测试之间按 group 隔离，构建时检查各部分有足够类别组。

本轮训练数据是既有真实正确鉴定与真实 trap，目的为验证特征增量；合成负样本 C/K/L 的增益要在固定特征后另作训练数据对照。当前使用的模型与训练模板为 `config/counterfactual/2da_label_dev_train.cv.yaml`，所有臂参数一致，旧模型不直接复用。

## 阈值和配对分析

复用 `cv_train.py` 与 `cv_core.py`：存储标签 1=正确、0=错误，模型输出 trust，评估时显式转为 `error_truth=1-label`、`error_score=1-trust`。错误鉴定是评估正类。

每个成员在自己的训练侧 OOF 分数上以正确 IDs 校准 FPR 1%、5%、10%，将阈值施加于对应成员的外层测试分数，最后五成员多数投票。正式结果位于各臂 `operating_points.*.external_ensemble`。名称中的 FPR5 指校准目标；外层测试实际 FPR 需另报，不能保证正好 5%。成员阈值不用于平均 ensemble trust。

汇总前检查：每行恰好被一个外层折评估、各臂测试身份一致、成员数量与列清单正确、平均 trust 与成员分数一致、阈值可从对应 OOF 分数复算、投票可从成员分数和阈值复算、混淆矩阵与保存结果一致。

报告 ROC-AUC、error PR-AUC、各锁定工作点实际 FPR/FNR/错误召回和 TP/FP/FN/TN。正式训练器的测试标签参与的回顾性 ROC 工作点仍只位于 `retrospective_test_working_points`，标记 oracle/non-deployable；主比较不用这些阈值。

主比较预先指定 B+QDS 对 B，B+Q/B+D/B+S 为探索性比较。以 `leakage_group_id` 为单位，在正确/错误组内分别有放回抽样，相同组的全部行及全部模型共用抽样权重，产生配对 95% percentile 区间。该原实验要求 class-pure connected groups，混合标签组会报错，不能拆成按行 bootstrap。

开发参考线沿用规划：错误召回点估计至少增加 3 个百分点，召回增量区间下界 >0，实际 FPR 增量区间上界 ≤0.5 个百分点。只有 B+QDS 的主比较可据此给出 `development_support`；单组不给优胜确认声明，其区间未作多重比较校正。记录所有原 FN→检出、原检出→新 FN、正确接受→新 FP、原 FP→正确接受及净找回数。

这里的“原 FN”是本次按统一协议重训的 B 的 FN。它不自动等于之前人工审阅的 130 条；新结果保留稳定 sample ID，可另行对应历史名单。人工审阅过的数据只构成开发验证，不能当作独立确认；bootstrap 只描述冻结预测的组间不确定性，不包含重新训练的不确定性，折间标准差也不是置信区间。

所有机器可读评估结果标记 `metric_semantics=error_identification_positive_v1` 和 `positive_class=incorrect_identification`。

## 输出

默认根目录：`runs/spec_trainer/single-peptide-structure-validation/`。

| 文件或目录 | 内容 |
| --- | --- |
| `report.md` | 实际 FPR/FNR、FN/FP 转移、配对区间及主比较结论 |
| `summary.csv` / `summary.json` | 五臂汇总；JSON 含完整工作点和分析约定 |
| `paired_comparisons.csv` | 主比较和三个探索比较、配对区间及净找回 |
| `transitions.csv` | 每行、每个增量臂对 B 的判定变化 |
| `group_transitions.csv` / `fold_metrics.csv` | 按 group 与外层折检查效果是否集中 |
| `pooled_test_predictions.csv` | 所有外层测试行的各臂平均 trust 和各工作点投票 |
| `training/fold_*/b*/` | 成员模型、OOF 分数、外层成员分数、阈值、日志 |
| `join_audit.json` / `protocol.json` | 合并覆盖、缺失、列清单、内层协议、输入与代码指纹 |
| `cohort.csv` / `folds/` / `configs/` | 冻结 cohort、各折训练测试输入和 25 个训练配置 |
| `source_outer_fold_manifest.csv` / `outer_fold_manifest.csv` | 原 manifest 副本与本次真实样本清单 |

## 验证

相关回归测试 105 项通过。另在 200 行测试夹具上实际运行 `make 2da-structure-validation`，完成 25 个训练任务、125 个成员模型和汇总，再续跑一次确认模型文件均未改写；含空格的输入/输出路径也通过。这些构造数据仅用于检查软件流程，不作为真实鉴定效果结果。

新增测试覆盖新行重排、漏行/重复、源身份错配、旧特征漂移、group/序列跨分组、非法折编号、缺失与可用性状态、输入冻结与防覆盖、早停/校准隔离、完整训练配置解析、锁与异常清理，以及完整的五折五臂正式 LightGBM 训练/汇总/续跑。

数值级测试固定：4 条正确鉴定中误报 1 条，则 FP=1、FPR=25%；4 条错误鉴定中漏放 2 条，则 FN=2、FNR=50%。另覆盖找回一条旧 FN 同时新增一条 FN 时净收益为 0，不能只报告找回数。

完整真实 manifest 的身份桥接也已用占位 Q/D/S 数据独立验算，确认 107,227 行与 38,192 个原分组可原样对应；这只验证合并逻辑，不是全量谱图提取或模型效果实验。实现测试数据上的指标不作为真实结果报告。
