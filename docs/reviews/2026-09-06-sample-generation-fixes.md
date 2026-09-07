样本生成流程修复与复验（2026-09-06）

本次修复对应 [初次检查的五项问题](2026-09-06-sample-generation-audit.md)，并完成两名 subagent 的独立复查。检查覆盖正例准备、三种 counterfactual 生成器、旧外部搜索生成路径、PSM 序列化、特征提取衔接、数据拼接、普通 CV、fixed-negpool、combined 和相关 deep-trainer 消费路径。

已完成的修复：

1. **预测谱覆盖与来源泄漏。** Pilot 关闭不覆盖新候选的原始预测谱库，增加 `ms1_ms2_no_prediction` + `evidence_observed` 的专用 CV 配置及 `make counterfactual-2da-train`。来源字段贯通 PSM 和 features；cohort 审计按来源记录筛选数量和预测覆盖率。合成训练禁止使用 `has_lib_pred`，使用预测特征或依赖预测覆盖的 cohort 时必须保证全部输入行有预测。
2. **父子样本跨折。** 抽取共享 `sample_groups.py`，在 cohort 筛选前连接 L/I 等价序列和 parent/peptide/family IDs。普通 CV 与 fixed-negpool 共用此逻辑，外层和内层切分使用相同连接组；直接 OOF API 会拒绝拆开已知家族的分组。OOF CSV 保留来源及家族元数据，结果 JSON 记录分组审计。
3. **L/I 等价候选重复。** 两条生成路径均在同一 parent 观测内按规范化序列去重，跨 raw 的独立观测保留；生成审计记录去重规则。
4. **旧坐标特征误拼接。** Counterfactual 拼接强制 query ID，并校验规范化 raw、RT、precursor m/z。仅容许 float32 序列化舍入误差，坐标变化时要求重提特征。保留旧外部搜索 manifest 的无观测坐标连接接口。
5. **raw-heldout 检查误报通过。** 所有非空训练输入及 heldout 的每行必须具备有效 raw 标识；支持逐行合并 raw 别名并拒绝冲突值。另行记录 L/I 规范化后的肽序列重叠，区分 domain holdout 与未见肽泛化。

Subagent 复查发现并补修了四个遗漏：

- 混用 raw 列名时，隔离检查后的去重仍可能误删独立观测。现于 assembly 入口统一 raw，直接去重接口也执行相同解析。
- 部分 child 缺少 query ID 时可能被 inner join 静默丢弃。现拒绝缺 ID 的 child 或来源不明行，仅允许明确的正例 parent 不带 query ID。
- 旧 `main.py` 按行训练入口可绕过家族保护。现拒绝合成样本输入，并指向 CV 入口。
- Fixed-negpool 筛选可能删除跨电荷父子连接中的 parent；combined 再分组也可能丢失该连接。现于筛选前建立连接，并在合并时保留已有连接组。

独立复验结果：生成与拼接侧相关测试 **70 passed**；训练、cohort、特征选择、fixed-negpool 和 deep-trainer 侧相关测试 **106 passed**。这些测试集合有重叠，不相加作为全仓通过数量。两名 subagent 最终均未发现仍未解决的可复现问题。

训练侧另外执行了真实 LightGBM 小型 cross-test：训练输入含 synthetic family，外部测试仅含 sequence/label/features，无 parent/group 元数据；训练和外部 ensemble 正常完成，序列重叠如实记录，固定工作点输出 `external_ensemble`。单域与三域实际 fixed API 也复验了“筛掉桥接 parent 后仍保持同组”。上述输入为受控测试数据，不代表真实科学实验效果。

最终全仓验证：**1,093 passed，6 skipped，1 warning，耗时 40.24 秒**。6 项跳过分别为 3 项缺真实 mzML、1 项缺真实 PFB、1 项缺既有 fold 模型、1 项缺 CUDA；未将跳过项计作通过。唯一警告来自已有的小样本五折回退测试。`git diff --check` 和训练 Makefile target 的 dry run 均通过。

```bash
PYTHONPATH=/tmp/ms2-met-audit-deps:$PWD \
MPLCONFIGDIR=/tmp/ms2-met-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m pytest -q -rs

make -n counterfactual-2da-train
```

兼容性与运行范围：

- 存储仍为 `label=1` 表示正确鉴定，模型输出 trust score；评估沿用 canonical error-positive helpers，没有更改 FP/FN/FPR/FNR 或 ensemble 决策定义，也没有改写历史模型或结果。
- 连接分组规则标为 `sequence_family_connected_components_v2`。旧候选、features 和分组结果不会自动迁移；对现有 counterfactual 产物应重新生成候选、重提特征，再运行新配置。专用 train target 只消费现有 feature snapshot，不隐式重建数据。
- 本机缺少配置指定的 target/contaminant FASTA、9 个 PFB 和谱库，因此未执行真实 5,000-parent pilot/XIC 重提取，也未测量真实负例产率、hardness 或外部泛化收益。
- 为执行此前缺依赖而跳过的模型测试，LightGBM 4.7.0 与 narwhals 2.25.0 安装在临时目录 `/tmp/ms2-met-audit-deps`，未改变项目依赖声明或系统安装。
