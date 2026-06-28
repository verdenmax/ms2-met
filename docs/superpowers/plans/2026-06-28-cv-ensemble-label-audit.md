# 5 折 CV + 折间 ensemble + 标签噪声审计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `spec_trainer` 的生产 LightGBM 增加 5 折分组 CV，用统一的 OOF 预测同时产出诚实评估、折间 ensemble 模型、与标签噪声审计名单。

**Architecture:** 新建两个文件——`cv_core.py`（纯函数：切折/指标/审计/集成均值，无 lightgbm 依赖、可单测）与 `cv_train.py`（CLI 编排：读 df、复用 `LGBModel` 逐折训练、拼 OOF、存 5 模型、写 cv.json 与 suspects.csv）。**不改 `main.py`**，现行单 holdout 流程不变。

**Tech Stack:** Python 3.11；LightGBM（经现有 `LGBModel`）；scikit-learn（`StratifiedGroupKFold`/`GroupShuffleSplit`）；pandas/numpy；pytest。运行需在装有 lightgbm+sklearn 的 conda 环境（如 `jianyan`，当前需先 `conda install -c conda-forge lightgbm scikit-learn`）。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `tools/spec_trainer/src/cv_core.py`（新建） | 纯函数、**不 import lightgbm**：`make_cv_splits` / `working_points` / `fnr_at_fpr5` / `evaluate_oof` / `audit_labels` / `average_proba`。可在无 lightgbm 下单测。 |
| `tools/spec_trainer/src/cv_train.py`（新建） | CLI + 编排：读 yaml、`pd.read_csv` 保留 sequence、`resolve_feature_cols`、`assemble_oof`（逐折训练/拼 OOF/存模型，经 `ModelManager`/`LGBModel`）、`predict_ensemble`、写 `cv.json`/`suspects.csv`。 |
| `tools/spec_trainer/config/cv_in_2da_clean.yaml`（新建） | 在 `in_2da_clean.yaml` 基础上加 CV 键；不动原配置（单 holdout 仍走原 yaml）。 |
| `tests/test_cv_core.py`（新建） | `cv_core` 单测（numpy/pandas/sklearn，**无 lightgbm**）。 |
| `tests/test_cv_train.py`（新建） | 端到端集成测试，`pytest.importorskip("lightgbm")`（缺 lightgbm 自动跳过）。 |
| `Makefile`（修改） | 新增 `train-cv-2da` 目标。 |

约定：`cv_core.py` 与 `cv_train.py` 同在 `tools/spec_trainer/src/`，测试用 `sys.path.insert(0, "tools/spec_trainer/src")`（与 `tests/` 里现有 spec_trainer 测试一致）。所有"缺陷/指标"函数纯净、可独立测。

---

## Task 1: `cv_core.make_cv_splits` —— 分组分层切折

**Files:**
- Create: `tools/spec_trainer/src/cv_core.py`
- Test: `tests/test_cv_core.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_cv_core.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "spec_trainer", "src"))


def test_make_cv_splits_grouped_no_leak_full_cover():
    from cv_core import make_cv_splits
    # 10 个肽段(group), 每个 2 行; group 与 label 一致(一条肽段同一类)
    groups = np.repeat(np.arange(10), 2)            # 0,0,1,1,...,9,9
    y = np.where(groups < 6, 1, 0)                  # 前 6 组正, 后 4 组负
    splits = make_cv_splits(y, groups, n_folds=5, seed=42)
    assert len(splits) == 5
    covered = np.concatenate([te for _, te in splits])
    assert sorted(covered.tolist()) == list(range(len(y)))   # 每行恰好测一次
    for tr, te in splits:
        assert set(groups[tr]).isdisjoint(set(groups[te]))   # 同肽段不跨 train/test


def test_make_cv_splits_no_groups_fallback():
    from cv_core import make_cv_splits
    y = np.array([1] * 16 + [0] * 4)
    splits = make_cv_splits(y, groups=None, n_folds=5, seed=42)
    assert len(splits) == 5
    covered = np.concatenate([te for _, te in splits])
    assert sorted(covered.tolist()) == list(range(len(y)))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_core.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'cv_core'`

- [ ] **Step 3: 写最小实现**

```python
# tools/spec_trainer/src/cv_core.py
"""Pure, lightgbm-free helpers for cross-validated training.

Split / metric / audit / ensemble-averaging logic lives here so it can be
unit-tested without importing lightgbm (mirrors the holdout.py / feature_cols.py
extraction pattern in this package).
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_cv_splits(y, groups, n_folds=5, seed=42):
    """Return a list of (train_idx, test_idx) positional index arrays.

    With groups: StratifiedGroupKFold — no group spans a fold's train+test
    (prevents same-peptide leakage). Without groups (None): StratifiedKFold.
    """
    y = np.asarray(y)
    dummy = np.zeros(len(y))
    if groups is not None:
        groups = np.asarray(groups)
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(dummy, y, groups))
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(dummy, y))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_core.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_core.py tests/test_cv_core.py
git commit -m "feat: add cv_core.make_cv_splits (grouped stratified CV)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: `cv_core.working_points` / `fnr_at_fpr5` —— FNR@FPR 指标（口径同 eval_baseline）

**Files:**
- Modify: `tools/spec_trainer/src/cv_core.py`
- Test: `tests/test_cv_core.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_cv_core.py
def test_working_points_and_fnr_clean_separation():
    from cv_core import working_points, fnr_at_fpr5
    neg = np.linspace(0.0, 0.99, 100)      # 负例分数 0..0.99
    pos = np.ones(100)                     # 正例分数全 1.0(完全高于负例)
    y = np.r_[np.ones(100), np.zeros(100)]
    s = np.r_[pos, neg]
    wp = working_points(y, s)
    # 阈值 = 负例 95 分位 (<1.0); 正例全部 >= 阈值 -> pos_recall=1
    assert wp["neg_recall_95"]["pos_recall"] == 1.0
    assert 0.93 <= wp["neg_recall_95"]["neg_recall"] <= 0.96
    assert fnr_at_fpr5(y, s) == 0.0
    assert set(wp) == {"neg_recall_95", "neg_recall_90", "neg_recall_80"}


def test_working_points_matches_eval_baseline():
    import importlib.util, numpy as np
    # 口径 parity: 与 tools/eval_baseline.py 同输出; 无法导入则跳过
    spec = importlib.util.find_spec("eval_baseline")
    if spec is None:
        import pytest
        pytest.skip("eval_baseline not importable in this env")
    from cv_core import working_points
    from eval_baseline import compute_working_points
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    assert working_points(y, s) == compute_working_points(y, s)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_core.py::test_working_points_and_fnr_clean_separation -q`
Expected: FAIL —— `ImportError: cannot import name 'working_points'`

- [ ] **Step 3: 写最小实现（追加到 cv_core.py）**

```python
def working_points(y_true, y_score, fpr_targets=(0.05, 0.10, 0.20)):
    """Negative-quantile working points — same convention as
    tools/eval_baseline.py:compute_working_points (FPR via neg quantile).
    Returns {"neg_recall_95/90/80": {threshold, pos_recall, neg_recall}}.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    out = {}
    for fpr in fpr_targets:
        thr = float(np.quantile(neg, 1 - fpr))
        pos_kept = int((pos >= thr).sum())
        neg_kept = int((neg >= thr).sum())
        out[f"neg_recall_{int((1 - fpr) * 100)}"] = {
            "threshold": thr,
            "pos_recall": float(pos_kept / max(len(pos), 1)),
            "neg_recall": float(1 - neg_kept / max(len(neg), 1)),
        }
    return out


def fnr_at_fpr5(y_true, y_score):
    """FNR at FPR<=5% = 1 - pos_recall at the neg-95% working point."""
    return 1.0 - working_points(y_true, y_score)["neg_recall_95"]["pos_recall"]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_core.py -q`
Expected: PASS（parity 测试在装有 spectrum/eval_baseline 的环境跑，否则 skip）

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_core.py tests/test_cv_core.py
git commit -m "feat: add cv_core.working_points + fnr_at_fpr5 (eval_baseline parity)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: `cv_core.evaluate_oof` —— OOF 汇总指标

**Files:**
- Modify: `tools/spec_trainer/src/cv_core.py`
- Test: `tests/test_cv_core.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_cv_core.py
def test_evaluate_oof_perfect_and_reversed():
    from cv_core import evaluate_oof
    y = np.r_[np.ones(50), np.zeros(50)]
    perfect = np.r_[np.full(50, 0.9), np.full(50, 0.1)]   # 正高负低
    m = evaluate_oof(y, perfect)
    assert m["auc"] == 1.0
    assert m["fnr_at_fpr5"] == 0.0
    assert "neg_recall_95" in m["working_points"]
    m2 = evaluate_oof(y, 1.0 - perfect)                   # 完全反向
    assert m2["auc"] == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_core.py::test_evaluate_oof_perfect_and_reversed -q`
Expected: FAIL —— `ImportError: cannot import name 'evaluate_oof'`

- [ ] **Step 3: 写最小实现（追加到 cv_core.py）**

```python
def evaluate_oof(y_true, oof_proba):
    """Summary metrics on out-of-fold predictions: auc + FNR@FPR5 + working points."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    oof_proba = np.asarray(oof_proba)
    return {
        "auc": float(roc_auc_score(y_true, oof_proba)),
        "fnr_at_fpr5": float(fnr_at_fpr5(y_true, oof_proba)),
        "working_points": working_points(y_true, oof_proba),
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_core.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_core.py tests/test_cv_core.py
git commit -m "feat: add cv_core.evaluate_oof (auc + fnr@fpr5 + working points)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: `cv_core.audit_labels` —— 标签噪声嫌疑名单

**Files:**
- Modify: `tools/spec_trainer/src/cv_core.py`
- Test: `tests/test_cv_core.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_cv_core.py
import pandas as pd


def test_audit_labels_negatives_only_sorted_filtered():
    from cv_core import audit_labels
    df = pd.DataFrame({
        "label": [0, 0, 0, 1, 1],
        "sequence": list("ABCDE"),
        "charge": [2, 2, 3, 2, 2],
        "all_p75": [0.9, 0.8, 0.1, 0.9, 0.9],
    })
    oof = [0.97, 0.92, 0.40, 0.99, 0.10]   # 第4个是正例(0.99)不应入榜
    susp = audit_labels(df, oof, threshold=0.9, top_n=10)
    assert list(susp["sequence"]) == ["A", "B"]            # 仅负例 A,B 过阈值, 降序
    assert "oof_proba" in susp.columns and "all_p75" in susp.columns
    assert list(susp["oof_proba"]) == [0.97, 0.92]
    assert len(audit_labels(df, oof, threshold=0.9, top_n=1)) == 1   # top_n 截断
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_core.py::test_audit_labels_negatives_only_sorted_filtered -q`
Expected: FAIL —— `ImportError: cannot import name 'audit_labels'`

- [ ] **Step 3: 写最小实现（追加到 cv_core.py）**

```python
def audit_labels(df, oof_proba, label_col="label", threshold=0.9, top_n=200,
                 id_cols=("sequence", "charge", "label_type"),
                 diag_cols=("all_p75", "precursor_pearson", "all_cosine_mean",
                            "all_heavy_shape_irregularity_max")):
    """Negatives ranked by how 'positive-looking' their OOF prob is.

    Triage list for manual review (NOT auto-relabel): a negative whose
    out-of-fold prob >= threshold either is a genuine hard negative or a
    mislabel. Returns id+diagnostic cols (only those present), oof desc,
    capped at top_n.
    """
    work = df.copy()
    work["oof_proba"] = np.asarray(oof_proba)
    neg = work[work[label_col] == 0]
    susp = (neg[neg["oof_proba"] >= threshold]
            .sort_values("oof_proba", ascending=False)
            .head(top_n))
    keep = ([c for c in id_cols if c in susp.columns]
            + ["oof_proba"]
            + [c for c in diag_cols if c in susp.columns])
    return susp[keep].reset_index(drop=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_core.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_core.py tests/test_cv_core.py
git commit -m "feat: add cv_core.audit_labels (OOF-based label-noise triage)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: `cv_core.average_proba` —— 折间 ensemble 均值

**Files:**
- Modify: `tools/spec_trainer/src/cv_core.py`
- Test: `tests/test_cv_core.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_cv_core.py
def test_average_proba():
    from cv_core import average_proba
    out = average_proba([np.array([0.0, 1.0]),
                         np.array([1.0, 0.0]),
                         np.array([0.5, 0.5])])
    assert np.allclose(out, [0.5, 0.5])
    assert np.allclose(average_proba([np.array([0.3, 0.7])]), [0.3, 0.7])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_core.py::test_average_proba -q`
Expected: FAIL —— `ImportError: cannot import name 'average_proba'`

- [ ] **Step 3: 写最小实现（追加到 cv_core.py）**

```python
def average_proba(proba_list):
    """Mean of per-fold predict_proba arrays = ensemble score for new data."""
    arr = np.vstack([np.asarray(p, dtype="f8") for p in proba_list])
    return arr.mean(axis=0)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_core.py -q`
Expected: PASS（全部 cv_core 测试通过）

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_core.py tests/test_cv_core.py
git commit -m "feat: add cv_core.average_proba (fold ensemble mean)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: CV 配置 + `cv_train.py` 骨架（路径派生，lightgbm 懒加载）

**Files:**
- Create: `tools/spec_trainer/config/cv_in_2da_clean.yaml`
- Create: `tools/spec_trainer/src/cv_train.py`
- Test: `tests/test_cv_train.py`

设计要点：`cv_train.py` **顶层不 import lightgbm**（`ModelManager` 在 `assemble_oof` 内懒加载），故无 lightgbm 也能 `import cv_train` 并单测路径派生。

- [ ] **Step 1: 写 CV 配置（复制 in_2da_clean.yaml + CV 键，不动原配置）**

```yaml
# tools/spec_trainer/config/cv_in_2da_clean.yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
  feature_cols: []
  target_col: label
  group_col: sequence          # StratifiedGroupKFold 按肽段分组防泄漏
model:
  type: lightgbm
  params:
    boosting_type: gbdt
    objective: binary
    metric: [auc, binary_logloss]
    num_leaves: 15
    learning_rate: 0.02
    feature_fraction: 0.9
    bagging_fraction: 0.8
    min_data_in_leaf: 50
    verbose: -1
training:
  num_boost_round: 1000
  early_stopping_rounds: 200
  cv_folds: 5
  cv_seed: 42
  valid_size: 0.15             # 折内早停验证集比例
audit:
  suspect_threshold: 0.9
  suspect_top_n: 200
output:
  model_path: runs/spec_trainer/models/cv_in_2da_clean.txt   # → .fold0..4.txt
  result_path: runs/spec_trainer/results/cv_in_2da_clean.cv.json
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_cv_train.py
import os, sys, json, importlib.util
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "spec_trainer", "src"))
from feature_cols import resolve_feature_cols

_HAS_LGB = importlib.util.find_spec("lightgbm") is not None
requires_lgb = pytest.mark.skipif(not _HAS_LGB, reason="lightgbm not installed")


def test_derive_paths():
    import cv_train                      # 须在无 lightgbm 下可导入
    cfg = {"output": {
        "model_path": "runs/m/cv_in_2da_clean.txt",
        "result_path": "runs/r/cv_in_2da_clean.cv.json"}}
    mp, rp, sp = cv_train.derive_paths(cfg)
    assert mp == "runs/m/cv_in_2da_clean"
    assert rp == "runs/r/cv_in_2da_clean.cv.json"
    assert sp == "runs/r/cv_in_2da_clean.cv.suspects.csv"


def test_read_dataframe_concat(tmp_path):
    import cv_train
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(a, index=False)
    pd.DataFrame({"x": [3]}).to_csv(b, index=False)
    df = cv_train.read_dataframe([str(a), str(b)])
    assert list(df["x"]) == [1, 2, 3]
```

- [ ] **Step 3: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_train.py::test_derive_paths -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'cv_train'`

- [ ] **Step 4: 写 cv_train.py 骨架**

```python
# tools/spec_trainer/src/cv_train.py
"""Cross-validated training entry for spec_trainer (production LightGBM).

One CV pass yields OOF predictions that drive: honest evaluation, fold
ensemble (saved per-fold models), and label-noise audit. Does NOT touch
main.py (single-holdout flow unchanged). lightgbm is imported lazily inside
assemble_oof so this module (and path helpers) import without it.
"""
import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd
import yaml

from cv_core import (average_proba, audit_labels, evaluate_oof, fnr_at_fpr5,
                     make_cv_splits)
from feature_cols import resolve_feature_cols


def read_dataframe(train_files):
    """Concatenate feature CSVs, keeping all columns (sequence/charge needed)."""
    return pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)


def derive_paths(cfg):
    """(model_prefix, result_path, suspects_path) from output.{model,result}_path."""
    model_path = cfg["output"]["model_path"]
    result_path = cfg["output"]["result_path"]
    model_prefix = re.sub(r"\.txt$", "", model_path)
    suspects_path = re.sub(r"\.json$", ".suspects.csv", result_path)
    return model_prefix, result_path, suspects_path


def _build_parser():
    p = argparse.ArgumentParser(description="CV train + ensemble + label audit")
    p.add_argument("--config", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--logpath", default="./cv_spec.log")
    return p
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_train.py::test_derive_paths tests/test_cv_train.py::test_read_dataframe_concat -q`
Expected: PASS（2 passed；这两个测试不需 lightgbm）

- [ ] **Step 6: 提交**

```bash
git add tools/spec_trainer/config/cv_in_2da_clean.yaml tools/spec_trainer/src/cv_train.py tests/test_cv_train.py
git commit -m "feat: cv_train skeleton + cv config (paths, lazy lightgbm)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 7: `cv_train.assemble_oof` —— 逐折训练 / 拼 OOF / 存模型

**Files:**
- Modify: `tools/spec_trainer/src/cv_train.py`
- Test: `tests/test_cv_train.py`

- [ ] **Step 1: 写失败测试（合成数据集成测试，缺 lightgbm 自动跳过）**

```python
# 追加到 tests/test_cv_train.py（_toy_df/_toy_cfg 为下方多个集成测试共用；
# resolve_feature_cols / requires_lgb 已在文件头部定义）


def _toy_df(n_groups=40, per=5, seed=0):
    """40 个肽段(group)×5 行；前 70% 组为正，all_p75 含信号。"""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        lab = 1 if g < int(n_groups * 0.7) else 0
        for _ in range(per):
            rows.append({"sequence": f"PEP{g}", "charge": 2, "label": lab,
                         "all_p75": rng.normal(lab, 1.0),           # 有信息
                         "precursor_pearson": rng.normal(0, 1.0)})  # 噪声
    return pd.DataFrame(rows)


def _toy_cfg(tmp_path):
    return {
        "data": {"feature_cols": [], "target_col": "label", "group_col": "sequence"},
        "model": {"type": "lightgbm", "params": {
            "objective": "binary", "num_leaves": 7, "learning_rate": 0.1,
            "min_data_in_leaf": 5, "verbose": -1}},
        "training": {"num_boost_round": 40, "early_stopping_rounds": 15,
                     "cv_folds": 5, "cv_seed": 42, "valid_size": 0.25},
        "audit": {"suspect_threshold": 0.5, "suspect_top_n": 50},
        "output": {"model_path": str(tmp_path / "m.txt"),
                   "result_path": str(tmp_path / "r.cv.json")},
    }


@requires_lgb
def test_assemble_oof_no_nan_and_saves_models(tmp_path):
    import cv_train
    df = _toy_df()
    csv = tmp_path / "feat.csv"; df.to_csv(csv, index=False)
    feature_cols = resolve_feature_cols(None, [str(csv)], "label")
    X = df[feature_cols]; y = df["label"]; groups = df["sequence"]
    cfg = _toy_cfg(tmp_path)
    oof, fold_metrics, model_paths = cv_train.assemble_oof(
        df, X, y, groups, cfg, feature_cols, str(tmp_path / "m"))
    assert not np.isnan(oof).any()                       # 每行恰好预测一次
    assert len(fold_metrics) == 5 and "auc" in fold_metrics[0]
    assert len(model_paths) == 5 and all(os.path.exists(p) for p in model_paths)
```

注：`_toy_df`/`_toy_cfg`/`requires_lgb`/`resolve_feature_cols` 供 Task 8、9 的集成测试复用。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_train.py::test_assemble_oof_no_nan_and_saves_models -q`
Expected: FAIL —— `AttributeError: module 'cv_train' has no attribute 'assemble_oof'`（或装了 lightgbm 时如此；未装则 skip）

- [ ] **Step 3: 写最小实现（追加到 cv_train.py）**

```python
def assemble_oof(df, X, y, groups, cfg, feature_cols, model_prefix):
    """Train one model per fold, collect leak-free OOF preds, save fold models.

    Returns (oof_proba, fold_metrics, model_paths). lightgbm imported here.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    from models.model_manager import ModelManager

    n_folds = int(cfg["training"].get("cv_folds", 5))
    seed = int(cfg["training"].get("cv_seed", 42))
    valid_size = float(cfg["training"].get("valid_size", 0.15))
    grp_vals = None if groups is None else groups.values

    splits = make_cv_splits(y.values, grp_vals, n_folds=n_folds, seed=seed)
    oof = np.full(len(y), np.nan)
    fold_metrics, model_paths = [], []

    for k, (tr_idx, te_idx) in enumerate(splits):
        # 折内早停验证集（分组，避免污染 OOF 折）
        if groups is not None:
            inner = GroupShuffleSplit(n_splits=1, test_size=valid_size,
                                      random_state=seed)
            loc_tr, loc_val = next(inner.split(
                X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]))
        else:
            inner = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                           random_state=seed)
            loc_tr, loc_val = next(inner.split(X.iloc[tr_idx], y.iloc[tr_idx]))
        tr2, val = tr_idx[loc_tr], tr_idx[loc_val]

        model = ModelManager.create(cfg, feature_names=feature_cols)
        model.fit(X.iloc[tr2], y.iloc[tr2], X.iloc[val], y.iloc[val])
        oof[te_idx] = model.predict_proba(X.iloc[te_idx])

        mp = f"{model_prefix}.fold{k}.txt"
        os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
        model.save(mp)
        model_paths.append(mp)

        te_y, te_p = y.iloc[te_idx].values, oof[te_idx]
        if len(set(te_y.tolist())) < 2:               # 该折单类 → auc 无定义
            fold_metrics.append({"fold": k, "auc": float("nan"),
                                 "fnr_at_fpr5": float("nan")})
        else:
            fold_metrics.append({"fold": k,
                                 "auc": float(roc_auc_score(te_y, te_p)),
                                 "fnr_at_fpr5": float(fnr_at_fpr5(te_y, te_p))})

    assert not np.isnan(oof).any(), "OOF has NaN — some sample never predicted"
    return oof, fold_metrics, model_paths
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_train.py -q`
Expected: PASS（装有 lightgbm 时）；否则集成测试 skip、骨架测试仍 PASS

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_train.py tests/test_cv_train.py
git commit -m "feat: cv_train.assemble_oof (per-fold train, OOF, save models)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 8: `cv_train.main` —— 编排：评估 + 审计 + 写出 cv.json / suspects.csv

**Files:**
- Modify: `tools/spec_trainer/src/cv_train.py`
- Test: `tests/test_cv_train.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_cv_train.py
@requires_lgb
def test_main_writes_outputs(tmp_path):
    import cv_train, yaml
    df = _toy_df()
    csv = tmp_path / "feat.csv"; df.to_csv(csv, index=False)
    cfg = _toy_cfg(tmp_path); cfg["data"]["train_files"] = [str(csv)]
    cfg_path = tmp_path / "cfg.yaml"; cfg_path.write_text(yaml.safe_dump(cfg))
    summary = cv_train.main(["--config", str(cfg_path), "--name", "toy",
                             "--logpath", str(tmp_path / "log.txt")])
    res = json.loads((tmp_path / "r.cv.json").read_text())
    assert "auc" in res and "fnr_at_fpr5" in res
    assert len(res["fold_metrics"]) == 5 and "auc_mean" in res
    assert (tmp_path / "r.cv.suspects.csv").exists()     # 派生路径
    assert summary["auc"] == res["auc"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_train.py::test_main_writes_outputs -q`
Expected: FAIL —— `AttributeError: module 'cv_train' has no attribute 'main'`（或装了 lightgbm 时如此；未装则 skip）

- [ ] **Step 3: 写最小实现（追加到 cv_train.py）**

```python
def main(argv=None):
    args = _build_parser().parse_args(argv)
    if os.path.dirname(args.logpath):
        os.makedirs(os.path.dirname(args.logpath), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(args.logpath, encoding="utf-8")])

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    target_col = cfg["data"]["target_col"]
    train_files = cfg["data"]["train_files"]
    df = read_dataframe(train_files)
    feature_cols = resolve_feature_cols(
        cfg["data"].get("feature_cols"), train_files, target_col)
    X = df[feature_cols]
    y = df[target_col]

    group_col = cfg["data"].get("group_col")
    if group_col and group_col in df.columns:
        groups = df[group_col]
    else:
        groups = None
        if group_col:
            logging.warning("group_col %r not in data — ungrouped CV", group_col)

    model_prefix, result_path, suspects_path = derive_paths(cfg)
    oof, fold_metrics, model_paths = assemble_oof(
        df, X, y, groups, cfg, feature_cols, model_prefix)

    summary = evaluate_oof(y, oof)
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
    summary.update({
        "cv_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "model_paths": model_paths,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    })
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)

    audit_cfg = cfg.get("audit", {})
    susp = audit_labels(
        df, oof, label_col=target_col,
        threshold=float(audit_cfg.get("suspect_threshold", 0.9)),
        top_n=int(audit_cfg.get("suspect_top_n", 200)))
    susp.to_csv(suspects_path, index=False)

    logging.info("CV done: AUC=%.4f FNR@FPR5=%.4f; %d suspects -> %s",
                 summary["auc"], summary["fnr_at_fpr5"], len(susp), suspects_path)
    return summary


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_train.py -q`
Expected: PASS（装有 lightgbm 时全过；否则集成测试 skip）

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_train.py tests/test_cv_train.py
git commit -m "feat: cv_train.main (OOF eval + label audit + write outputs)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 9: `cv_train.predict_ensemble` —— 折间 ensemble 打分（消费侧）

**Files:**
- Modify: `tools/spec_trainer/src/cv_train.py`
- Test: `tests/test_cv_train.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_cv_train.py
@requires_lgb
def test_predict_ensemble_in_range(tmp_path):
    import cv_train
    df = _toy_df()
    csv = tmp_path / "feat.csv"; df.to_csv(csv, index=False)
    feature_cols = resolve_feature_cols(None, [str(csv)], "label")
    X = df[feature_cols]; y = df["label"]; groups = df["sequence"]
    cfg = _toy_cfg(tmp_path)
    _, _, model_paths = cv_train.assemble_oof(
        df, X, y, groups, cfg, feature_cols, str(tmp_path / "m"))
    s = cv_train.predict_ensemble(model_paths, X.values)
    assert s.shape == (len(X),)
    assert (s >= 0).all() and (s <= 1).all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cv_train.py::test_predict_ensemble_in_range -q`
Expected: FAIL —— `AttributeError: module 'cv_train' has no attribute 'predict_ensemble'`（未装 lightgbm 则 skip）

- [ ] **Step 3: 写最小实现（追加到 cv_train.py）**

```python
def predict_ensemble(model_paths, X):
    """Ensemble score for NEW data = mean of the per-fold models' predictions.

    X: numpy array or DataFrame with the same feature columns/order used in
    training (lightgbm matches by position). Used to score external data
    (e.g. cross_test, production) — NOT for in-sample eval (use OOF for that).
    """
    import lightgbm as lgb
    probas = [lgb.Booster(model_file=p).predict(X) for p in model_paths]
    return average_proba(probas)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cv_train.py -q`
Expected: PASS（装有 lightgbm 时）

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_train.py tests/test_cv_train.py
git commit -m "feat: cv_train.predict_ensemble (mean of fold models)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 10: Makefile 目标 `train-cv-2da`

**Files:**
- Modify: `Makefile`（训练区，`.PHONY: train-... train-all` 行附近，见 Makefile:855）

- [ ] **Step 1: 加 .PHONY 声明**

在 `Makefile:855` 的 `.PHONY: train-legacy-all train-clean-all ... train-all` 行**之后**新增一行：

```makefile
.PHONY: train-cv-2da
```

- [ ] **Step 2: 加目标（放在 `train-clean-all:` 目标之后，约 Makefile:996 之后）**

```makefile
# 5 折分组 CV + 折间 ensemble + 标签审计（生产 LightGBM；见
# docs/superpowers/specs/2026-06-28-cv-ensemble-label-audit-design.md）
train-cv-2da: runs/baseline_2da_clean/features.csv
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results
	$(PY) tools/spec_trainer/src/cv_train.py \
	    --config tools/spec_trainer/config/cv_in_2da_clean.yaml \
	    --name cv_in_2da_clean \
	    --logpath runs/spec_trainer/cv_spec.log
	@echo "[done] CV → runs/spec_trainer/results/cv_in_2da_clean.cv.json (+ .suspects.csv)"
```

- [ ] **Step 3: 验证目标展开（无需数据/依赖）**

Run: `make -n train-cv-2da PY="conda run -n jianyan python"`
Expected: 打印出 `conda run -n jianyan python tools/spec_trainer/src/cv_train.py --config tools/spec_trainer/config/cv_in_2da_clean.yaml ...`（dry-run，不实跑）

- [ ] **Step 4: 提交**

```bash
git add Makefile
git commit -m "build: add train-cv-2da target (CV + ensemble + label audit)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## 端到端实跑（需真实数据 + 依赖，人工 checkpoint，非 CI）

> 与 spec §11 一致：本步需 `conda install -n jianyan -c conda-forge lightgbm scikit-learn` 且 `runs/baseline_2da_clean/features.csv` 为**新 schema**（含 shape-defect 列；若仍是旧 131 列需先重抽特征，见上一里程碑）。非阻塞单测。

- [ ] 跑 `make train-cv-2da PY="conda run -n jianyan python"`。
- [ ] 验 `cv_in_2da_clean.cv.json`：`auc / fnr_at_fpr5 / auc_mean±auc_std / fold_metrics(5)` 合理；与单 holdout 的 `in_2da_clean.json` 比对（CV 的 FNR@FPR 应更稳）。
- [ ] 看 `cv_in_2da_clean.cv.suspects.csv`：人工复核 top 嫌疑负例是真·hard 负例还是标错（人源同源/共享肽/FDR 误判）。
- [ ] 记入 checkpoint。

---

## Self-Review（已核对，记录于此）

- **Spec 覆盖**：§3.1 CV→Task1/6/7；§3.2 ensemble→Task5/9；§3.3 审计→Task4/8；§4 指标(复用 working_points)→Task2/3；§5 配置→Task6；§6 产出→Task8;§7 边界(单类折 NaN/分组缺失回退/OOF 无 NaN 断言)→Task7/8；§8 测试→各 Task TDD + Task7-9 集成；§9 不动 main.py→全程；§11 环境→端到端步骤。✅ 全覆盖。
- **占位符**：无 TBD/TODO；每个代码步给完整代码与命令。✅
- **类型/命名一致**：`make_cv_splits/working_points/fnr_at_fpr5/evaluate_oof/audit_labels/average_proba`（cv_core）与 `read_dataframe/derive_paths/assemble_oof/main/predict_ensemble`（cv_train）在各 Task 引用处签名一致；`assemble_oof(df,X,y,groups,cfg,feature_cols,model_prefix)→(oof,fold_metrics,model_paths)` 三处一致；`requires_lgb`/`_toy_df`/`_toy_cfg` 定义在前、引用在后。✅
- **依赖隔离**：cv_core 不 import lightgbm；cv_train 顶层不 import lightgbm（`ModelManager` 懒加载于 `assemble_oof`），故 Task6 骨架测试无 lightgbm 也能跑；lightgbm 测试用 `@requires_lgb` 跳过。✅
