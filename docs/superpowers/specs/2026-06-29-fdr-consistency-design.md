# FDR 一致性分析:1% 模型在 1~50% 正例上的召回衰减 设计

**日期**:2026-06-29
**背景**:用代谢标记 MS2 训好的 CV 集成模型(1% clean,`cv_in_2da_clean.fold0..4`)是搜索引擎之外的独立证据。想用它检验不同置信度的正例:把正例按 q-value 放宽到 50%,看每个 FDR 箱里有多少仍被模型判为"有真实代谢支撑"。1% 正例(高置信)应几乎全召回;FDR 越高,召回应越低=高 FDR 正例渐多假阳,缺代谢支撑。这是搜索引擎评测(用途2)的独立交叉验证。

**目的**:量化"正例召回 vs FDR"衰减曲线,并核验(a) 1-5% 箱是否与 1% 一致、(b) 5 折模型在各箱是否一致(mean±std)。

**关键代码**:复用现成 5 折模型 + cv_core 指标;新增独立分析脚本 + 一份高 FDR 正例数据。不改 cv_train/cv_core/main。

---

## 1. 数据(新抽,远程)
1. `extract_2da_pos50.ini`(复制 `extract_2da_pfind_diann.ini`,正例 `qvalue_threshold=0.50`,负例仍 1%)→ `datasets/hela_2da_pos50.json`。
2. `runs/baseline_2da_pos50/config.ini`(复制 `baseline_2da_clean/config.ini`,指向 pos50 JSON;输出 `runs/baseline_2da_pos50/features.csv`)。
3. **保留 q-value**:提取管线把每条 PSM 的 q-value 写入 features.csv 一列 `q_value`(原现有列无此项);加入 `feature_cols.EXCLUDED_EXTRA`,**绝不进训练**(仅分箱用)。
4. Makefile `pos50-2da` 目标(extract → features.csv)。

## 2. 分析脚本 `tools/fdr_consistency.py`
- 载 `runs/spec_trainer/models/cv_in_2da_clean.fold0..4.txt`(5 折)+ pos50 features.csv + 1% clean features.csv;
- 阈值:1% clean 正负例 → `eval_baseline.compute_working_points` 的 neg_recall_95 工作点(FNR@FPR≤5%);
- 正例按 `q_value` 分箱:`(0,0.01] (0.01,0.05] (0.05,0.10] (0.10,0.20] (0.20,0.50]`;
- 每箱:ensemble 召回(`avg≥thr` 占比)+ 5 折各自召回 → mean±std;
- 出 `runs/spec_trainer/results/fdr_consistency_2da.csv`(bin/n/ensemble_recall/fold_mean/fold_std) + 衰减曲线图(rsvg/matplotlib,1% 基线)。

## 3. 测试
合成 5 折模型 + 带 q_value 合成正例:断言分箱边界、recall 计算、mean±std、阈值口径同 eval_baseline。`q_value` 在 EXCLUDED_EXTRA 的契约测试。脚本可无真实数据单测(`@requires_lgb` 仅集成部分)。

## 4. 兼容/不在范围
不改训练/cv_core;不回填历史;仅 2da 起,5da/normal 后续。q_value 仅分箱,不改模型。
