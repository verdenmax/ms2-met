# FDR 一致性分析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 1% clean 训好的 5 折 CV 集成模型,对正例 FDR≤50% 的数据按 q-value 分箱算召回衰减(+5 折 mean±std),独立验证高 FDR 正例是否仍有代谢标记支撑。

**Architecture:** 复用现成 `cv_in_2da_clean.fold0..4` 模型;新增独立分析脚本 `tools/fdr_consistency.py`;提取侧把每条 PSM 的 `q_value`(已在 PSM 链路)写入 features.csv 并加进 `EXCLUDED_EXTRA` 防泄漏;新增 pos50 提取配置 + make 目标。不改 cv_train/cv_core/main。

**Tech Stack:** Python 3.11;LightGBM Booster(加载 .txt 折模型);pandas/numpy/pyyaml/scipy;pytest。测试用 `conda run -n jianyan python -m pytest`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `workflows/flow_utils.py`(改) | 两处 meta-row 加 `q_value: psm._q_value`。 |
| `tools/spec_trainer/src/feature_cols.py`(改) | `EXCLUDED_EXTRA` 加 `q_value`(分箱列,绝不进训练)。 |
| `extract_2da_pos50.ini`(新) | 正例 qvalue 0.50,出 hela_2da_pos50.json。 |
| `runs/baseline_2da_pos50/config.ini`(新) | 指向 pos50 json,出 features.csv。 |
| `tools/fdr_consistency.py`(新) | 分箱召回 + ensemble + 5折 mean±std → csv + 图。 |
| `tests/test_fdr_consistency.py`(新) | 纯函数单测(分箱/recall/mean±std)。 |
| `Makefile`(改) | `pos50-2da` 目标。 |

---

## Task 1: features.csv 写入 q_value + 防泄漏排除

**Files:** Modify `workflows/flow_utils.py:108,141`; Modify `tools/spec_trainer/src/feature_cols.py:36`; Test `tests/test_feature_cols_contract.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fdr_consistency_excl.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "spec_trainer", "src"))
from feature_cols import EXCLUDED_EXTRA
def test_q_value_excluded():
    assert "q_value" in EXCLUDED_EXTRA
```

- [ ] **Step 2:** `conda run -n jianyan python -m pytest tests/test_fdr_consistency_excl.py -q` → FAIL

- [ ] **Step 3:** `feature_cols.py` EXCLUDED_EXTRA 加 `"q_value",`;`flow_utils.py` 两 dict 在 `"label_type": ...` 后加 `"q_value": psm._q_value,`(single 用 psm,multi 用 psm1)。

- [ ] **Step 4:** pytest → PASS

- [ ] **Step 5:** commit `feat: write q_value to features.csv, exclude from training`

---

## Task 2: pos50 提取配置 + make 目标

**Files:** Create `extract_2da_pos50.ini`, `runs/baseline_2da_pos50/config.ini`; Modify `Makefile`

- [ ] **Step 1:** 建 `extract_2da_pos50.ini`(复制 extract_2da_pfind_diann.ini,`result_file=./datasets/hela_2da_pos50.json`,pfind/diann `qvalue_threshold=0.50`,无 negative_qvalue 行)。
- [ ] **Step 2:** 建 `runs/baseline_2da_pos50/config.ini`(复制 baseline_2da_clean/config.ini,`light_result_file=./datasets/hela_2da_pos50.json`,`pfind_qvalue_threshold=0.50`,`result_file=./runs/baseline_2da_pos50/features.csv`)。
- [ ] **Step 3:** Makefile 加(real TAB):
```makefile
.PHONY: pos50-2da
pos50-2da: datasets/hela_2da_pos50.json runs/baseline_2da_pos50/config.ini
	$(PY) main.py --configpath runs/baseline_2da_pos50/config.ini --logpath runs/baseline_2da_pos50/extract.log
datasets/hela_2da_pos50.json: extract_2da_pos50.ini
	$(PY) tools/extract_common.py --configpath extract_2da_pos50.ini
```
- [ ] **Step 4:** `make -n pos50-2da PY="conda run -n jianyan python"` → 展开正确;commit。

## Task 3: `tools/fdr_consistency.py`(分箱召回 + ensemble + 5折 mean±std)

**Files:** Create `tools/fdr_consistency.py`; Test `tests/test_fdr_consistency.py`

- [ ] **Step 1: 失败测试**
```python
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, "tools")
from fdr_consistency import bin_recall, FDR_BINS
def test_bin_recall_mean_std():
    df = pd.DataFrame({"q_value":[0.005,0.03,0.03,0.3],"label":[1]*4})
    fold_p = [np.array([.9,.9,.1,.1]),np.array([.9,.1,.1,.1])]
    r = bin_recall(df, fold_p, thr=0.5)
    assert r[(0,0.01)]["ens_recall"]==1.0
    assert r[(0.2,0.5)]["ens_recall"]==0.0
    assert abs(r[(0.01,0.05)]["fold_mean"]-0.75)<1e-9
```
- [ ] **Step 2:** pytest → FAIL
- [ ] **Step 3:** 实现 `FDR_BINS=[(0,.01),(.01,.05),(.05,.10),(.10,.20),(.20,.50)]`;`bin_recall(df,fold_probas,thr)`:每箱 mask=q∈(lo,hi],ens=mean(folds)≥thr 召回,fold_mean/std=各折≥thr 召回;`main()` 加载 5 折 Booster+pos50/clean csv,clean 算 thr,出 fdr_consistency_2da.csv+图。
- [ ] **Step 4:** pytest → PASS;commit

## Task 4: 自检
- spec 覆盖:q_value(T1)/pos50数据(T2)/分箱召回曲线(T3)。占位无。命名 bin_recall/FDR_BINS 一致。

## 端到端(远程):`make pos50-2da` → `python tools/fdr_consistency.py` → 看 csv/图。
