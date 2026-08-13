# Deep trainer

本目录承载代谢标记 DIA 校验的深度学习实验。第一阶段是现有
`evidence_all` 表格特征上的小型 PyTorch MLP；第二阶段直接读取轻/重前体与
碎片离子的原始 XIC，以卷积编码器和 masked fragment attention 自动学习信号
表示。两阶段共用同一个冻结 E20 测试集和指标协议。

## 为什么先做表格 MLP

当前 LightGBM 已经是很强的基线。在导出原始 XIC 张量之前，先只改变模型，
保持以下口径完全相同：

- `evidence_all` 特征；
- `evidence_common` 共同队列；
- E5/E10/E20 嵌套负样本审计；
- 从已完成的 LightGBM bundle 读取、而非重新生成的固定 E20 测试集；
- 相同 outer folds 和 inner early-stopping folds；
- 与 LightGBM 相同，以 ROC-AUC 作为 early-stopping 指标；
- `cv_core.py` 的错误鉴定阳性指标；
- 固定测试决策使用每个成员的 outer-OOF 阈值和多数投票；
- 同时运行 logistic regression，并与冻结 LightGBM 进行配对 bootstrap。

这里的 E20 是**内部、固定、按 sequence（有 family ID 时按连通 family）分组的
holdout**，不是独立 external entrapment test。旧特征快照没有 `group_id`、
`pair_id` 或 `candidate_family_id` 时，结果会显式报告只能保证 sequence 不泄漏。

如果小 MLP 没有稳定超过 LightGBM，说明下一阶段应重点增加原始 XIC 信息，
而不是单纯扩大表格分类器。

## 运行

推荐使用根目录 Make target：

```bash
make train-deep-mlp-combined \
  FEATURE_ROOT=/path/to/feature_result/ms2-met-runs-08-08
```

必须先对同一特征快照生成冻结 LightGBM 协议：

```bash
make train-fixed-test-negpool-combined \
  FEATURE_ROOT=/path/to/feature_result/ms2-met-runs-08-08
```

MLP 会校验九个输入 CSV 的完整 SHA256、LightGBM 冻结的有序特征列表、全部
`sample_id` 和 fold map；任意不一致都会停止。LightGBM 协议和深度学习结果都先
在同级临时目录完成整包构建，再原子发布；失败的 `--overwrite` 会保留旧结果。
旧的冻结协议若没有 `feature_cols_sha256`，需要用当前代码重新运行一次
`train-fixed-test-negpool-combined`。

默认只训练 M20。若要同时训练 M5/M10/M20，复制配置并修改：

```yaml
experiment:
  negative_pool_models: [M5, M10, M20]
```

缺失值捷径敏感性实验使用同一个 `DEEP_PROTOCOL_ROOT`，只替换配置和输出根目录：

```bash
make train-deep-mlp-combined \
  FEATURE_ROOT="$FEATURE_ROOT" \
  DEEP_CONFIG=tools/deep_trainer/config/tabular_mlp_no_missing_indicators.yaml \
  DEEP_OUTPUT_ROOT=runs/deep_trainer/no-missing-indicators
```

两臂必须使用相同的 Python/PyTorch/NumPy/Pandas 版本。完成有、无缺失指示的
两次运行后，用下面的命令进行逐样本、按 sequence 成簇的配对 bootstrap；比较器
会拒绝测试样本或上述运行环境不一致的结果：

```bash
python -m tools.deep_trainer.missingness_sensitivity \
  --without-indicators-root \
    runs/deep_trainer/no-missing-indicators/tabular-mlp/combined \
  --with-indicators-root \
    runs/deep_trainer/with-indicators/tabular-mlp/combined \
  --output-dir runs/deep_trainer/missingness-sensitivity-comparison
```

直接运行时，需要先生成与 LightGBM 相同的 split config：

```bash
python tools/spec_trainer/gen_cv_configs.py \
  --feature-root "$FEATURE_ROOT" \
  --output-root runs/deep_trainer/reference-cv \
  --config-dir runs/deep_trainer/configs \
  --feature-arm evidence_all

python -m tools.deep_trainer.experiment \
  --config tools/deep_trainer/config/tabular_mlp.yaml \
  --split-config runs/deep_trainer/configs/cv_in_2da_neg20.yaml \
  --feature-root "$FEATURE_ROOT" \
  --protocol-root runs/spec_trainer/fixed-negpool/combined \
  --dataset combined \
  --output-root runs/deep_trainer/tabular-mlp/combined
```

先只验证数据协议、不训练：在第二个命令追加 `--prepare-only`。它会生成一个
状态为 `prepare_only` 的独立 bundle；正式训练应使用不同输出目录，或明确加
`--overwrite`。

## 输出

- `preflight.json`：数据、队列、嵌套负样本和固定切分审计；
- `config_used.yaml`、`split_config_used.yaml`：本次实际使用的冻结配置；
- `manifests/`：逐行 membership、固定测试集清单和 sequence/family fold 映射；
- `models/*.pt`：每折模型、fold-local 预处理器和特征名；
- `predictions/*_train_oof.csv`：训练集严格 OOF 分数；
- `predictions/fixed_test_predictions.csv`：固定测试集 ensemble/vote 分数；
- `fixed_test_summary.csv`：与现有 LightGBM 表格同口径的核心指标；
- `paired_model_bootstrap.csv`：MLP/logistic/LightGBM 的配对置信区间；
- `missingness_audit.csv`：按 label 和数据域统计的缺失模式审计；
- `domain_summary.csv`：2da/5da/normal 分域结果；
- `summary.json`：模型、指标语义、fold 指标和 provenance。

默认 `device: auto` 会优先使用 CUDA，没有可用 GPU 时退回 CPU。第一轮可以先
减少 `epochs` 或只保留一个 seed；正式 XIC 模型训练建议使用 GPU。

## Phase 2：原始 XIC 数据层

Phase 2 不直接把模型扩大，而是先导出轻重前体、重标 M+1/M+2 同位素包络和
按 `b/y × ordinal × fragment charge` 区分的 MS2 XIC。第一步只构建
`3 × (200 correct + 200 incorrect) = 1200` 条完整性 pilot：

```bash
make build-deep-xic-pilot \
  FEATURE_ROOT=/path/to/feature_result/ms2-met-runs-08-08 \
  DEEP_PROTOCOL_ROOT=/path/to/fixed-negpool/combined
```

默认使用 `workspace/<dataset>/*.dia.npz`；可通过
`PHASE2_CACHE_ROOT=/path/to/cache` 覆盖。Phase 2 缓存同时绑定数据域、配置中的
raw 绝对路径、文件大小、mtime_ns 和内容 SHA256；旧缓存缺少这些字段时，只有
raw 仍可访问才会重建，不能在 cache-only 模式下被静默复用。三个
`baseline_*_neg20/config.ini` 必须位于 `FEATURE_ROOT`，其中的 PSM JSON、raw
路径、标记化学、ppm 和 XIC 窗口是本次提取的来源。

构建器首先验证冻结协议的 SHA256 与 sample ID，然后在每个数据域、每个存储
标签内进行确定性均衡抽样。序列、蛋白、label type、negative tier、数据域、
绝对 RT、前体电荷和划分信息只写入 Parquet manifest 供审计，不是允许的模型
输入。没有谱库时仍输出所有样本，并用 `fragment_prediction_present=0` 表示。

成功输出包括：

- `manifest.parquet`：sample ID、冻结 split/fold 及审计元数据；
- `shards/*/*.npy`：可 mmap 的前体固定张量和 ragged 碎片张量；
- `schema.json`：通道、mask、状态码和模型输入白名单；
- `audit/identity_matching.csv`：PSM JSON 到冻结 sample ID 的唯一匹配；
- `audit/feature_parity.csv`：从保存张量回算现有特征的逐值比较；
- `checksums.json` 与 `COMPLETE`：parity 从已落盘的 mmap shard 回算；只有全部
  检查通过才原子发布。`COMPLETE` 保存 checksum 清单本身的 SHA256，读取器默认
  校验全部文件，覆盖发布中断后会恢复唯一的旧版本备份。

输出 schema 为 `phase2_raw_xic_v2`，并把当前同位素模型写入顶层 contract。
若冻结 E20 snapshot 的 `isotope_model` 不是当前的
`ideal_full_label_exact_mass_v2`（或未声明），`isotope_correlation` 仍逐值写入
parity 审计，但标为
`legacy_isotope_model_audit_only`，不作为发布阻断项；这是算法迁移差异，不是
shard 完整性错误。其余前体/碎片 parity 始终强制。用当前代码重新提取特征并
重建冻结协议后，该同位素字段自动恢复为强制 parity。

pilot 只用于完整性检查，不能进入正式训练。pilot parity 通过后，全量构建使用：

```bash
make build-deep-xic-full \
  FEATURE_ROOT=/path/to/feature_result/ms2-met-runs-08-08 \
  DEEP_PROTOCOL_ROOT=/path/to/fixed-negpool/combined \
  PHASE2_CACHE_ROOT=/path/to/cache
```

全量构建严格覆盖冻结协议的全部 train/test sample ID。它按 raw 逐个加载、按
m/z panel 一次扫描一张 MS1/MS2 谱图、流式写入 shard；中断后再次执行同一命令
会读取 `.building/RESUME_STATE.json`，只复用已经原子提交且来源指纹完全一致的
shard。配置、划分、raw、PSM、DIA cache、冻结协议，或任一张量生成模块的源码
内容有变化时拒绝 resume，避免不同算法的 shard 混入同一数据集。启用谱库预测
时，pepdata、RT、MS2 prediction、FASTA 和 modification 文件也全部记录 SHA256。

原始 DIA cache 仍采用兼容旧流程的压缩 `.npz`；全量 Phase 2 会为每个 raw 一次性
原子生成同级 `.mmap-v1/` 目录，将成员拆成独立 `.npy`。正式提取从该目录以真正
的只读 `np.memmap` 打开大数组，不会再把压缩 NPZ 的全部数组误称为 mmap。代价是
需要额外缓存磁盘空间；源 NPZ 指纹变化时 mmap 目录自动重建。

## Phase 2：XIC 深度模型训练

全量 XIC 的 `COMPLETE` 存在后运行：

```bash
make train-deep-xic-combined \
  FEATURE_ROOT=/path/to/feature_result/ms2-met-runs-08-08 \
  DEEP_PROTOCOL_ROOT=/path/to/fixed-negpool/combined \
  PHASE2_FULL_XIC_OUTPUT_ROOT=/path/to/phase2-xic/full
```

默认配置是 `phase2/config/xic_fusion.yaml`，训练 M20 的 3 个随机种子 × 5 个冻结
outer folds，共 15 个成员。每个成员只在该折 inner-training rows 拟合，在冻结的
inner-validation rows 上用 ROC-AUC early stop，并对自己的 outer-OOF rows 校准
FPR 1%/5%/10% error-score 阈值。固定测试集的正式决策是 15 个成员分别应用各自
OOF 阈值后多数投票；不会把单成员阈值直接应用到平均 ensemble score。

模型输入只有：

- 轻/重前体及重标 M+1/M+2 的 intensity、ppm error、相对 RT、scan/peak mask；
- 成对轻/重碎片 XIC，以及 ion type、fragment charge；
- 从同一条 XIC 推导的强度/时间尺度。

`label_type`、q-value、negative tier、dataset、绝对 RT、sequence、split/fold 和
蛋白信息只参与划分或审计，不进入网络。谱库预测强度默认关闭；只有单独构建了
`prediction.include=true` 的 XIC 数据集并把模型配置显式改为 true 才能启用，
因此无谱库数据仍能完整训练。

不可分离的同 m/z 轻重碎片、heavy 超出采集范围以及未实际尝试配对的碎片，只
保留在审计张量中，attention 权重强制为 0。`fragment_ordinal` 和显式 fragment
count 也不进入网络，避免肽长或理论碎片数量成为构造负例的捷径。

网络由一个前体 1D-CNN、所有 fragment 共享的 1D-CNN、离子属性 embedding、
masked attention set pooling 和融合 MLP 组成，输出仍是
`trust_score=P(correct identification)`。统计边界统一转换为错误鉴定阳性，所有
ROC、PR、FNR 和固定 FPR 指标复用 `spec_trainer/src/cv_core.py`。

正式运行前只做协议检查：

```bash
python -m tools.deep_trainer.phase2.experiment \
  --config tools/deep_trainer/phase2/config/xic_fusion.yaml \
  --split-config runs/deep_trainer/configs/cv_in_2da_neg20.yaml \
  --feature-root "$FEATURE_ROOT" \
  --protocol-root runs/spec_trainer/fixed-negpool/combined \
  --signal-root runs/deep_trainer/phase2-xic/full \
  --output-root runs/deep_trainer/phase2-xic-preflight \
  --prepare-only
```

训练结果包括 fold checkpoint、严格 OOF 分数、固定测试逐样本分数、15 成员多数
投票工作点、分数据域指标，以及相同测试行/相同 leakage group 上相对冻结
LightGBM M20 的 paired cluster bootstrap。Checkpoint 绑定 XIC 数据集的 checksum
identity 和精确输入归一化/mask adapter contract；换了 shard 内容或输入适配规则
后不会静默推理。结果 bundle 也带 `checksums.json` 与 `COMPLETE`，覆盖发布中断时
会恢复旧 bundle。
