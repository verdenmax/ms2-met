# Deep trainer

本目录承载代谢标记 DIA 校验的深度学习实验。第一阶段是现有
`evidence_all` 表格特征上的小型 PyTorch MLP；原始 XIC CNN、fragment
attention 和 light/heavy 双塔将在同一实验接口下继续扩展。

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

当前机器没有可用 CUDA，但默认 `device: auto` 会在 CPU 上运行。第一轮可以
先减少 `epochs` 或只训练 M20；正式 XIC 双塔训练建议使用 GPU。
