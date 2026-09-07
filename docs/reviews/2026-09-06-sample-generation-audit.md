样本生成流程检查记录（2026-09-06）

这是修复前的检查记录，基准 HEAD 为 `c88c73a`；下文的复现和行号描述当时的代码。随后用户授权修复，五项问题及 subagent 复查发现的遗漏均已处理，详见 [修复与复验记录](2026-09-06-sample-generation-fixes.md)。

覆盖路径为：搜索结果解析与 `extract_common` 正例筛选 → `counterfactual_parents` → 三种 counterfactual 候选生成器 → PSM JSON/manifest → 普通单 raw 特征提取 → `training_set_builder` 拼接与训练切分。另检查了旧版“生成 FASTA 后外部重新搜索”的生成、拼接接口，以及相关测试和设计文档。

结论：标签和质量计算的主要约定在所检查链路中保持一致，但存在两项应在训练前处理的高优先级衔接风险，以及三项已复现的数据完整性问题。下面区分最小复现的结论和真实数据尚未验证的范围。

1. **[P1] 原始预测谱库与合成候选不配套，会丢掉候选或暴露其来源。**

   位置：`runs/counterfactual_2da_label_dev_train/config.ini:32`，`workflows/pair_flow.py:194`，`workflows/single_work.py:965`，`tools/spec_trainer/src/cohort.py:9`，`tools/spec_trainer/src/feature_cols.py:107`。

   pilot 直接沿用原始蛋白库的预测谱；`PredStore` 只按候选自身的序列、修饰和电荷查找，不会为新生成序列补算预测谱。不命中的候选得到 `has_lib_pred=0`、`n_fragments_in_F=0` 和缺失的预测谱特征。

   接入现有正式配置的 `evidence_common` 队列时，`has_lib_pred=1` 是硬门槛，这些合成候选会被删除。改用旧式自动特征选择则会把 `has_lib_pred` 等列选入训练；即使单独删除该标志，整组预测特征的缺失模式仍可透露候选来源。

   最小复现：从 12 个真实 prepared parent 构造 48 个 shuffle 子例，再使用仅含 parent 预测的模拟 `PredStore`。任务分发得到 12 个正例全部命中、48 个负例全部不命中；其余 eligibility 字段均满足时，`evidence_common` 抛出“只剩 12 个 correct、0 个 error”的异常。旧式特征解析实际选择了 `has_lib_pred`、`n_fragments_in_F`、`spec_pattern_SA`。

   这验证了处理机制。当前机器无法访问真实谱库，尚未测量实际候选的命中比例；不能把模拟中的 100% 未命中当作真实结果。

   修复方向：为全部候选生成一致的预测谱；或者为这次实验定义不依赖谱库覆盖的特征集合与队列，同时排除预测特征及其缺失代理。训练前按样本来源记录谱库覆盖率和队列删除数量。

2. **[P1] 普通 CV 入口没有强制检查父子关系，沿用 `group_col: sequence` 会跨折。**

   位置：`tools/spec_trainer/src/cv_train.py:133`、`:589`，`tools/spec_trainer/config/cv_in_2da_clean.yaml:13`。

   生成器正确写出了 `parent_id`、`group_id` 和跨电荷、跨 L/I 的 `peptide_group_id`。但普通 CV 入口只使用配置指定的一列，并不检查该列是否把一个 parent family 拆开。现有基线配置使用 `sequence`；改写序列后的 child 与 parent 自然属于不同组。当前 counterfactual Makefile 终止于特征提取，也没有配套的训练配置防止这一误用。

   最小复现：选取 prepared 数据中的 12 个不同 peptide group，以模拟 target index 生成 60 行 parent/child 特征。`_validate_frame(..., group_col='sequence')` 接受输入；按当前 CV 使用的 5 折算法、seed=42 切分，各折训练集与验证集重叠的 parent family 数分别为 **5、7、7、8、8**。这是切分复现，没有训练模型，也没有测量指标虚高的幅度。

   `fixed_negpool._assign_leakage_groups` 已有连接序列与父子元数据的实现；该保护目前没有自动应用于普通 `cv_train` 入口。修复可复用这套分组能力，并验证所选分组包含完整 `peptide_group_id`；同时提供专用 counterfactual 训练配置。不能仅依赖 CSV 中存在分组列。

3. **[P2] 同一 parent 内只按原始字符串去重，L/I 等价候选会重复计数。**

   位置：`tools/counterfactual_negatives.py:394`、`:633`；旧版生成器 `tools/training_set_builder.py:355`、`:567` 也使用字符串去重。

   候选与 parent 的差异检查、target/contaminant 排除均考虑 L/I 等价性，但已生成候选集合保存的是未经规范化的原始序列。因而两个质谱上等价的负例可以获得不同 `query_id`，甚至被归到不同生成来源。

   已复现：parent=`GAILLLLK`，seed=11，composition 和 K/R-position 各生成 2 条时，同时输出 `GLLLLAIK` 和 `GLLLIALK`。两者 L/I 规范化后均为 `GLLLLALK`，却分别归入 composition 与 K/R-position 来源。二者继承同一 raw、RT、m/z、电荷，并具有相同的理论质量与 SILAC 位点分布。

   影响是重复加权同一假设、夸大候选多样性，并污染来源比较。修复应在 parent 观测内按 L/I 规范化序列去重；跨 raw 的独立观测仍需保留。

4. **[P2] 新 manifest 可以静默匹配到旧 RT 坐标提取的特征。**

   位置：`tools/counterfactual_negatives.py:611`，`tools/training_set_builder.py:1131`、`:1147`。

   counterfactual `query_id` 由 parent、raw、来源、候选序列和电荷构成，不包含 RT 或 precursor m/z。拼接特征时虽然优先使用 `query_id`，但只核对 `sequence`、`charge`，没有携带、比较 manifest 中的 raw/RT/m/z。输入 PSM 坐标更新后，旧特征与新 manifest 可以保持相同 ID 并通过连接。

   已复现：对相同 parent 先在 RT=12.5 生成一次，再把 parent RT 改成 20.0 生成一次；两次生成的 query ID 完全相同。把第一次特征与第二次 manifest 连接，函数成功返回 RT=12.5 的旧特征，未报告当前 manifest 的 RT=20.0。

   修复应校验规范化 raw 标识、RT、m/z 及提取输入版本。浮点坐标需使用明确的序列化容差；发现不一致时拒绝拼接，而不是重贴 provenance。

5. **[P2] 留出集隔离检查在训练输入缺失 raw 字段时仍报告通过。**

   位置：`tools/training_set_builder.py:654`、`:659`。

   `_check_heldout_disjoint` 要求 heldout 存在 raw 列，却静默忽略没有 raw 列的训练表；已有列中的缺失值也被 `dropna()` 忽略。因此缺少完整观测来源的数据仍可能被记录为已验证无泄漏。

   最小复现：传入一行无 raw 列的训练表，以及一行有 raw 的 heldout 表，即使 `require_heldout=True`，返回结果仍是 `checked=true, n_train_raws=0, n_heldout_raws=1`。

   修复应要求每张非空训练表、heldout 表的每一行都有有效 raw 标识，再执行交集检查。对于 domain holdout，还应另行记录肽序列重叠，避免将其解释成未见肽泛化。

已检查且未发现上述类型问题的部分：

- 正例准备按上游过滤 JSON 的 `label_type=positive` 接受真值，检查 raw split、修饰、序列、末端、电荷和坐标，并分配 L/I 规范化的 peptide group。这是当前设计明确选择的真值来源；`heavy_confirmed=True` 在这里是准备契约标记，不代表本次又读取谱图做了一次独立确认。
- composition shuffle 保留末端和组成；K/R-position shuffle 固定 K/R；local mass-gap 重新检查 precursor 兼容性及理论碎片差异。local 目前使用理论质量差，没有 observed fragment anchors，也没有在生成时完成 hardness 判定；文档已明确这一限制。
- 子例保留 parent 的观测上下文，heavy precursor 与 fragment shifts 按子例自身序列重新计算；没有把 parent 的轻重标质量差强行复制给改变 K/R 的 child。
- JSON 到单 raw 特征输出保存父子关系、split 和标签。存储约定仍为正确鉴定 `label=1`、错误鉴定 `label=0`。评估边界使用错误鉴定为正类的 canonical helpers；本次未更改任何指标实现或历史结果。
- 已有测试验证单进程/多进程候选、manifest 和行顺序一致，pilot 抽样集合不受输入顺序影响。计数、shortfall 和 runtime 配置有审计记录。

本地数据与验证范围：

- 现有 prepared 数据有 **103,755** 条正例、**35,910** 个 peptide group、**9** 个 raw。PSM sidecar 摘要通过校验；parent audit 记录的上游 JSON、raw-split CSV 摘要与当前文件一致。
- 对全部 prepared parent 计算未修饰理论 precursor 的一致性：无一超过配置的 20 ppm，最大绝对差约 0.07732 ppm。这是坐标与理论质量的一致性检查，不是对原始谱峰质量误差或鉴定真值的独立验证。
- 本地没有配置所指向的 target FASTA、contaminant FASTA、预测谱库，以及 9 个 PFB。因此没有执行完整 5,000-parent pilot、真实 XIC 重提取或 LightGBM 训练；真实负例产率、谱库覆盖率、hardness 和外部泛化仍未验证。
- 运行了两组相关现有测试：第一组 **65 passed, 2 skipped**；第二组 **246 passed, 7 skipped**，合计 **311 passed, 9 skipped**。9 项因未安装 LightGBM 而跳过；第二组有一条小样本分层切分警告。未把 skipped 计作通过。
- 最小复现脚本和输出保存在本机临时目录 `/tmp/ms2-met-sample-audit/`。其中 target index 和谱库覆盖复现使用显式模拟输入；真实 parent 只用于测试分组与生成接口。

执行的测试集合：

```text
第一组
tests/test_counterfactual_parents.py
tests/test_counterfactual_negatives.py
tests/test_counterfactual_pipeline_config.py
tests/test_training_set_builder.py
tests/test_psm_info_compat.py
tests/test_sample_identity.py
tests/test_training_cohort.py
tests/test_fixed_negpool.py

第二组
tests/test_extract_common.py
tests/test_extract_common_dual_fdr.py
tests/test_extract_common_integration.py
tests/test_extract_contaminant.py
tests/test_extract_label_site.py
tests/test_entrapment_classifier.py
tests/test_pair_flow_dispatch.py
tests/test_pair_flow_accounting.py
tests/test_label_derivation.py
tests/test_single_work_numerics.py
tests/test_feature_postfilter.py
tests/test_pred_pipeline_integration.py
tests/test_pred_pipeline_corner.py
tests/test_feature_cols_contract.py
tests/test_feature_groups.py
tests/test_cv_train.py
tests/test_cv_core.py
```
