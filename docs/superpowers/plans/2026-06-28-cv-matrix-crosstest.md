# CV 全矩阵 + cross_test 模式 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 spec_trainer 的 CV 从单实验推广到完整 30 矩阵(15 in-sample + 15 cross_test),并给 `cv_train.py` 加 cross_test 模式(训练侧 CV 出 5 模型 → ensemble 预测外部测试集 → 在外部测试集评估/审计)。

**Architecture:** 唯一"硬"代码改动是 `cv_train.py`:新增 `evaluate_cross_test` helper(复用 `predict_ensemble`/`average_proba`/`fnr_at_fpr5`)+ `main()` 加 `test_files` 分支。其余是 `gen_cv_configs.py` 生成 30 个 `cv_*.yaml`(从现有 in_/cross_test 配置变换而来,committed)+ Makefile `train-cv-*-all`/`train-cv-all` 目标。不改 `main.py`/`holdout.py`/`cv_core.py`/既有 30 配置。

**Tech Stack:** Python 3.11;LightGBM(经 `LGBModel`/`Booster`);scikit-learn;pandas/numpy/pyyaml;pytest。测试在 `jianyan` conda 环境跑(`conda run -n jianyan python -m pytest ...`,含 sklearn/lightgbm/pyyaml)。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `tools/spec_trainer/src/cv_train.py`(改) | 新增 `evaluate_cross_test(model_paths, X, y)` helper;`main()` 加 in_sample/cross_test 分支。 |
| `tools/spec_trainer/gen_cv_configs.py`(新) | 纯函数 `to_cv_config(src, name)`(in_/cross_test 源 dict → cv_ 变体 dict)+ `main()` 批量生成 30 个 `cv_*.yaml`。无 lightgbm 依赖。 |
| `tools/spec_trainer/config/cv_*.yaml`(新,30 个,生成并 commit) | 30 个 CV 配置(`cv_in_*` 15 + `cv_cross_test_*` 15)。 |
| `Makefile`(改) | `CV_*_YAMLS` 变量 + `train-cv-clean-all`...`train-cv-neg20-all` + `train-cv-all`。 |
| `tests/test_cv_train.py`(改) | `evaluate_cross_test` 单测 + cross_test 模式集成测试(`@requires_lgb`)。 |
| `tests/test_gen_cv_configs.py`(新) | `to_cv_config` 变换单测(无 lightgbm)。 |

约定:测试用 `conda run -n jianyan python -m pytest`;`cv_train.py` 保持顶层无 lightgbm(`evaluate_cross_test` 内 lazy import)。

---

## Task 1: `cv_train.evaluate_cross_test` —— 外部测试集的 per-fold + ensemble 评估

**Files:**
- Modify: `tools/spec_trainer/src/cv_train.py`
- Test: `tests/test_cv_train.py`

- [ ] **Step 1: 写失败测试(集成,`@requires_lgb`)**

```python
# 追加到 tests/test_cv_train.py（_toy_df/_toy_cfg/requires_lgb/resolve_feature_cols 已在文件头）
@requires_lgb
def test_evaluate_cross_test(tmp_path):
    import cv_train, lightgbm as lgb
    dfA = _toy_df(seed=0)               # 训练数据集 A
    dfB = _toy_df(seed=7)               # 外部测试数据集 B（不同抽样）
    csvA = tmp_path / "a.csv"; dfA.to_csv(csvA, index=False)
    feature_cols = resolve_feature_cols(None, [str(csvA)], "label")
    cfg = _toy_cfg(tmp_path)
    _, _, model_paths = cv_train.assemble_oof(
        dfA, dfA[feature_cols], dfA["label"], dfA["sequence"],
        cfg, feature_cols, str(tmp_path / "m"))
    Xb = dfB[feature_cols].values
    yb = dfB["label"].values
    ens, per_fold, agg = cv_train.evaluate_cross_test(model_paths, Xb, yb)
    assert ens.shape == (len(dfB),)
    # ensemble = 各折预测的均值
    per = [lgb.Booster(model_file=p).predict(Xb) for p in model_paths]
    assert np.allclose(ens, np.mean(per, axis=0))
    assert len(per_fold) == 5 and "auc" in per_fold[0] and "fnr_at_fpr5" in per_fold[0]
    assert {"test_auc_mean", "test_auc_std",
            "test_fnr_at_fpr5_mean", "test_fnr_at_fpr5_std"} <= set(agg)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n jianyan python -m pytest tests/test_cv_train.py::test_evaluate_cross_test -q`
Expected: FAIL —— `AttributeError: module 'cv_train' has no attribute 'evaluate_cross_test'`

- [ ] **Step 3: 写最小实现(追加到 cv_train.py,放在 `predict_ensemble` 之后)**

```python
def evaluate_cross_test(model_paths, X, y):
    """Score external test data with each fold model + the ensemble.

    Returns (ens_proba, per_fold, agg):
    - ens_proba: mean of the K fold models' predictions on X (cross_test score).
    - per_fold: [{fold, auc, fnr_at_fpr5}, ...] for each fold model on the
      external test (NaN when y is single-class).
    - agg: {test_auc_mean/std, test_fnr_at_fpr5_mean/std} over non-NaN folds.
    lightgbm imported lazily.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    probas = [lgb.Booster(model_file=p).predict(X) for p in model_paths]
    ens = average_proba(probas)
    one_class = len(set(y.tolist())) < 2
    per_fold = []
    for k, p in enumerate(probas):
        if one_class:
            per_fold.append({"fold": k, "auc": float("nan"),
                             "fnr_at_fpr5": float("nan")})
        else:
            per_fold.append({"fold": k,
                             "auc": float(roc_auc_score(y, p)),
                             "fnr_at_fpr5": float(fnr_at_fpr5(y, p))})
    aucs = [m["auc"] for m in per_fold if not np.isnan(m["auc"])]
    fnrs = [m["fnr_at_fpr5"] for m in per_fold if not np.isnan(m["fnr_at_fpr5"])]
    agg = {
        "test_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "test_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "test_fnr_at_fpr5_mean": float(np.mean(fnrs)) if fnrs else float("nan"),
        "test_fnr_at_fpr5_std": float(np.std(fnrs)) if fnrs else float("nan"),
    }
    return ens, per_fold, agg
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n jianyan python -m pytest tests/test_cv_train.py -q`
Expected: PASS(集成测试 RAN,不 skip)

- [ ] **Step 5: 确认 cv_train 仍无顶层 lightgbm**

Run: `conda run -n jianyan python -c "import sys; sys.modules['lightgbm']=None; sys.path.insert(0,'tools/spec_trainer/src'); import cv_train; print('import OK')"`
Expected: 打印 `import OK`(lightgbm 仅在 `evaluate_cross_test`/`assemble_oof`/`predict_ensemble` 内 lazy)

- [ ] **Step 6: 提交**

```bash
git add tools/spec_trainer/src/cv_train.py tests/test_cv_train.py
git commit -m "feat: cv_train.evaluate_cross_test (per-fold + ensemble on external test)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: `cv_train.main` 加 in_sample / cross_test 分支

**Files:**
- Modify: `tools/spec_trainer/src/cv_train.py`(替换整个 `main()` 函数体)
- Test: `tests/test_cv_train.py`

- [ ] **Step 1: 写失败测试(cross_test 模式集成,`@requires_lgb`)**

```python
# 追加到 tests/test_cv_train.py
@requires_lgb
def test_main_cross_test_mode(tmp_path):
    import cv_train, yaml
    dfA = _toy_df(seed=0)                # 训练数据集 A
    dfB = _toy_df(seed=7)               # 外部测试数据集 B
    a = tmp_path / "a.csv"; dfA.to_csv(a, index=False)
    b = tmp_path / "b.csv"; dfB.to_csv(b, index=False)
    cfg = _toy_cfg(tmp_path)
    cfg["data"]["train_files"] = [str(a)]
    cfg["data"]["test_files"] = [str(b)]            # 触发 cross_test
    cfg_path = tmp_path / "cfg.yaml"; cfg_path.write_text(yaml.safe_dump(cfg))
    summary = cv_train.main(["--config", str(cfg_path), "--name", "xt",
                             "--logpath", str(tmp_path / "log.txt")])
    res = json.loads((tmp_path / "r.cv.json").read_text())
    assert res["mode"] == "cross_test"
    assert len(res["test_per_fold"]) == 5
    assert "test_auc_mean" in res and "train_oof_auc" in res
    assert res["n_pos"] == int((dfB["label"] == 1).sum())   # 计数来自 B
    assert (tmp_path / "r.cv.suspects.csv").exists()
    assert summary["auc"] == res["auc"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n jianyan python -m pytest tests/test_cv_train.py::test_main_cross_test_mode -q`
Expected: FAIL —— `KeyError: 'mode'`(现 `main` 不写 `mode`,且不处理 test_files)

- [ ] **Step 3: 用下面完整版替换现有 `main()` 函数体**

(保留 `main` 之前所有函数与 `main` 之后的 `predict_ensemble`/`evaluate_cross_test`/`if __name__` 不变;只替换 `def main(argv=None):` 到 `return summary` 这一段。)

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

    test_files = cfg["data"].get("test_files")
    if test_files and set(test_files) != set(train_files):
        # cross_test 模式：ensemble 给外部测试集打分，在外部测试集评估/审计
        test_df = read_dataframe(test_files)
        eval_df = test_df
        eval_y = test_df[target_col]
        ens_proba, test_per_fold, test_agg = evaluate_cross_test(
            model_paths, test_df[feature_cols].values, eval_y.values)
        eval_proba = ens_proba
        summary = evaluate_oof(eval_y, eval_proba)
        train_oof = evaluate_oof(y, oof)
        summary.update({
            "mode": "cross_test",
            "test_per_fold": test_per_fold,
            "train_oof_auc": train_oof["auc"],
            "train_oof_fnr_at_fpr5": train_oof["fnr_at_fpr5"],
            "train_fold_metrics": fold_metrics,
        })
        summary.update(test_agg)
    else:
        # in_sample 模式（行为同上一里程碑）
        eval_df = df
        eval_y = y
        eval_proba = oof
        summary = evaluate_oof(eval_y, eval_proba)
        aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
        fnrs = [m["fnr_at_fpr5"] for m in fold_metrics
                if not np.isnan(m["fnr_at_fpr5"])]
        summary.update({
            "mode": "in_sample",
            "fold_metrics": fold_metrics,
            "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std": float(np.std(aucs)) if aucs else float("nan"),
            "fnr_at_fpr5_mean": float(np.mean(fnrs)) if fnrs else float("nan"),
            "fnr_at_fpr5_std": float(np.std(fnrs)) if fnrs else float("nan"),
        })

    summary.update({
        "cv_folds": len(fold_metrics),
        "model_paths": model_paths,
        "n_pos": int((eval_y == 1).sum()),
        "n_neg": int((eval_y == 0).sum()),
        "name": args.name,
    })
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)

    audit_cfg = cfg.get("audit", {})
    susp = audit_labels(
        eval_df, eval_proba, label_col=target_col,
        threshold=float(audit_cfg.get("suspect_threshold", 0.9)),
        top_n=int(audit_cfg.get("suspect_top_n", 200)))
    susp.to_csv(suspects_path, index=False)

    logging.info("CV(%s) done: AUC=%.4f FNR@FPR5=%.4f; %d suspects -> %s",
                 summary["mode"], summary["auc"], summary["fnr_at_fpr5"],
                 len(susp), suspects_path)
    return summary
```

- [ ] **Step 4: 跑新测试 + in_sample 回归测试**

Run: `conda run -n jianyan python -m pytest tests/test_cv_train.py -q`
Expected: PASS —— `test_main_cross_test_mode` 通过,且现有 `test_main_writes_outputs`(in_sample)仍通过(它断言的 `auc/fnr_at_fpr5/fold_metrics/auc_mean/fnr_at_fpr5_mean/name` 在 in_sample 分支仍写出;新增的 `mode` 不影响其 `in` 断言)。

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/src/cv_train.py tests/test_cv_train.py
git commit -m "feat: cv_train.main cross_test branch (ensemble-score external test set)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: `gen_cv_configs.py` —— in_/cross_test 配置 → cv_ 变体生成器

**Files:**
- Create: `tools/spec_trainer/gen_cv_configs.py`
- Test: `tests/test_gen_cv_configs.py`

- [ ] **Step 1: 写失败测试(纯函数,无 lightgbm)**

```python
# tests/test_gen_cv_configs.py
import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "spec_trainer"))


def _src_in():
    return {"data": {"train_files": ["runs/baseline_2da_clean/features.csv"],
                     "feature_cols": [], "target_col": "label", "test_size": 0.2},
            "model": {"type": "lightgbm", "params": {"num_leaves": 15}},
            "training": {"num_boost_round": 1000, "early_stopping_rounds": 200,
                         "valid_size": 0.2},
            "output": {"model_path": "runs/spec_trainer/models/in_2da_clean.txt",
                       "result_path": "runs/spec_trainer/results/in_2da_clean.json",
                       "figures_dir": "runs/spec_trainer/figures"}}


def test_to_cv_config_in_sample():
    from gen_cv_configs import to_cv_config
    cv = to_cv_config(_src_in(), "in_2da_clean")
    assert cv["data"]["group_col"] == "sequence"
    assert cv["training"]["cv_folds"] == 5 and cv["training"]["cv_seed"] == 42
    assert cv["training"]["valid_size"] == 0.15
    assert cv["audit"] == {"suspect_threshold": 0.9, "suspect_top_n": 200}
    assert cv["output"]["model_path"] == "runs/spec_trainer/models/cv_in_2da_clean.txt"
    assert cv["output"]["result_path"] == "runs/spec_trainer/results/cv_in_2da_clean.cv.json"
    assert "figures_dir" not in cv["output"]
    assert cv["data"]["train_files"] == _src_in()["data"]["train_files"]   # 保留
    assert cv["training"]["num_boost_round"] == 1000                       # 保留


def test_to_cv_config_cross_test_preserves_test_files():
    from gen_cv_configs import to_cv_config
    src = {"data": {"train_files": ["runs/baseline_5da_clean/features.csv",
                                    "runs/baseline_normal_clean/features.csv"],
                    "test_files": ["runs/baseline_2da_clean/features.csv"],
                    "feature_cols": [], "target_col": "label", "test_size": 0.0},
           "model": {"type": "lightgbm", "params": {"num_leaves": 15}},
           "training": {"num_boost_round": 1000, "early_stopping_rounds": 200,
                        "valid_size": 0.2},
           "output": {"model_path": "runs/spec_trainer/models/cross_test_2da_clean.txt",
                      "result_path": "runs/spec_trainer/results/cross_test_2da_clean.json",
                      "figures_dir": "x"}}
    cv = to_cv_config(src, "cross_test_2da_clean")
    assert cv["data"]["test_files"] == src["data"]["test_files"]           # 保留
    assert cv["output"]["result_path"] == \
        "runs/spec_trainer/results/cv_cross_test_2da_clean.cv.json"


def test_to_cv_config_does_not_mutate_source():
    from gen_cv_configs import to_cv_config
    src = _src_in()
    before = copy.deepcopy(src)
    to_cv_config(src, "in_2da_clean")
    assert src == before                                                   # deepcopy
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n jianyan python -m pytest tests/test_gen_cv_configs.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'gen_cv_configs'`

- [ ] **Step 3: 写实现**

```python
# tools/spec_trainer/gen_cv_configs.py
"""Generate cv_*.yaml variants from the existing in_*/cross_test_* configs.

Each cv variant adds CV keys (group_col / cv_folds / cv_seed / inner valid_size
/ audit) and rewrites output paths to cv-specific files (so CV results never
collide with the single-holdout main.py outputs). Run once and commit the
generated files; re-run to resync after the source configs change. Idempotent.
"""
import copy
import glob
import os

import yaml

_CFG_DIR = os.path.join(os.path.dirname(__file__), "config")


def to_cv_config(src, name):
    """Transform an in_/cross_test source config dict into its cv_ variant.

    name: source basename (e.g. 'in_2da_clean'); the variant is 'cv_' + name.
    Does NOT mutate src (deep-copies). train_files/test_files/model/target_col
    are preserved; CV keys + cv output paths are set; figures_dir is dropped.
    """
    cfg = copy.deepcopy(src)
    cfg["data"]["group_col"] = "sequence"
    cfg["training"]["cv_folds"] = 5
    cfg["training"]["cv_seed"] = 42
    cfg["training"]["valid_size"] = 0.15
    cfg["audit"] = {"suspect_threshold": 0.9, "suspect_top_n": 200}
    cv_name = "cv_" + name
    cfg["output"] = {
        "model_path": f"runs/spec_trainer/models/{cv_name}.txt",
        "result_path": f"runs/spec_trainer/results/{cv_name}.cv.json",
    }
    return cfg


def main():
    srcs = sorted(glob.glob(os.path.join(_CFG_DIR, "in_*.yaml"))
                  + glob.glob(os.path.join(_CFG_DIR, "cross_test_*.yaml")))
    for path in srcs:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            src = yaml.safe_load(f)
        cv = to_cv_config(src, name)
        out = os.path.join(_CFG_DIR, f"cv_{name}.yaml")
        with open(out, "w") as f:
            yaml.safe_dump(cv, f, sort_keys=False, allow_unicode=True)
    print(f"generated {len(srcs)} cv configs in {_CFG_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n jianyan python -m pytest tests/test_gen_cv_configs.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add tools/spec_trainer/gen_cv_configs.py tests/test_gen_cv_configs.py
git commit -m "feat: gen_cv_configs (in_/cross_test -> cv_ config transform)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: 生成并提交 30 个 `cv_*.yaml`

**Files:**
- Create(生成): `tools/spec_trainer/config/cv_*.yaml`(30 个)

- [ ] **Step 1: 运行生成器**

Run: `conda run -n jianyan python tools/spec_trainer/gen_cv_configs.py`
Expected: 打印 `generated 30 cv configs in .../config`

- [ ] **Step 2: 校验数量与关键字段**

Run:
```bash
ls tools/spec_trainer/config/cv_*.yaml | wc -l
conda run -n jianyan python -c "
import yaml
a=yaml.safe_load(open('tools/spec_trainer/config/cv_in_2da_clean.yaml'))
b=yaml.safe_load(open('tools/spec_trainer/config/cv_cross_test_2da_clean.yaml'))
assert a['data']['group_col']=='sequence' and a['training']['cv_folds']==5
assert a['output']['result_path']=='runs/spec_trainer/results/cv_in_2da_clean.cv.json'
assert 'figures_dir' not in a['output']
assert b['data']['test_files']==['runs/baseline_2da_clean/features.csv']
assert b['output']['result_path']=='runs/spec_trainer/results/cv_cross_test_2da_clean.cv.json'
print('config spot-check OK')
"
```
Expected: `30` 然后 `config spot-check OK`

- [ ] **Step 3: 提交 30 个配置**

```bash
git add tools/spec_trainer/config/cv_*.yaml
git commit -m "feat: generate 30 cv_*.yaml configs (in_ + cross_test full matrix)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: Makefile `train-cv-*-all` + `train-cv-all`

**Files:**
- Modify: `Makefile`(在现有 `train-cv-2da` 目标之后插入下面整块;变量 `$(CLEAN_FEATURES)`/`$(NEG05_FEATURES)`/.../`$(NEG20_FEATURES)` 已在文件前部 ~line 934 定义,直接复用)

- [ ] **Step 1: 在 `train-cv-2da` 目标之后插入以下整块**

```makefile
# ---------- CV 全矩阵(in-sample + cross_test ensemble)----------
.PHONY: train-cv-clean-all train-cv-neg05-all train-cv-neg10-all
.PHONY: train-cv-neg15-all train-cv-neg20-all train-cv-all

CV_CLEAN_YAMLS := cv_in_2da_clean cv_in_5da_clean cv_in_normal_clean \
                  cv_cross_test_2da_clean cv_cross_test_5da_clean cv_cross_test_normal_clean
CV_NEG05_YAMLS := $(subst _clean,_neg05,$(CV_CLEAN_YAMLS))
CV_NEG10_YAMLS := $(subst _clean,_neg10,$(CV_CLEAN_YAMLS))
CV_NEG15_YAMLS := $(subst _clean,_neg15,$(CV_CLEAN_YAMLS))
CV_NEG20_YAMLS := $(subst _clean,_neg20,$(CV_CLEAN_YAMLS))

train-cv-clean-all: $(CLEAN_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results
	@for y in $(CV_CLEAN_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config tools/spec_trainer/config/$$y.yaml --name $$y \
		    --logpath runs/spec_trainer/cv_spec.log || exit 1; \
	done
	@echo "[done] train-cv-clean-all (6 CV experiments)"

train-cv-neg05-all: $(NEG05_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results
	@for y in $(CV_NEG05_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config tools/spec_trainer/config/$$y.yaml --name $$y \
		    --logpath runs/spec_trainer/cv_spec.log || exit 1; \
	done
	@echo "[done] train-cv-neg05-all (6 CV experiments)"

train-cv-neg10-all: $(NEG10_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results
	@for y in $(CV_NEG10_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config tools/spec_trainer/config/$$y.yaml --name $$y \
		    --logpath runs/spec_trainer/cv_spec.log || exit 1; \
	done
	@echo "[done] train-cv-neg10-all (6 CV experiments)"

train-cv-neg15-all: $(NEG15_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results
	@for y in $(CV_NEG15_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config tools/spec_trainer/config/$$y.yaml --name $$y \
		    --logpath runs/spec_trainer/cv_spec.log || exit 1; \
	done
	@echo "[done] train-cv-neg15-all (6 CV experiments)"

train-cv-neg20-all: $(NEG20_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results
	@for y in $(CV_NEG20_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config tools/spec_trainer/config/$$y.yaml --name $$y \
		    --logpath runs/spec_trainer/cv_spec.log || exit 1; \
	done
	@echo "[done] train-cv-neg20-all (6 CV experiments)"

train-cv-all:
	$(MAKE) train-cv-clean-all
	$(MAKE) train-cv-neg05-all
	$(MAKE) train-cv-neg10-all
	$(MAKE) train-cv-neg15-all
	$(MAKE) train-cv-neg20-all
	@echo "[done] train-cv-all finished (30 CV experiments)"
```

注意:所有 recipe 行用**真实 TAB** 缩进(`\` 续行用 tab+空格),与文件其余部分一致。

- [ ] **Step 2: 校验展开(无需数据/依赖)**

Run:
```bash
# for 循环 recipe 在 make -n 下只回显一次,故数 for 循环里的配置名(应 6)
make -n train-cv-clean-all 2>&1 | grep -oE "cv_(in|cross_test)_[a-z0-9_]+" | sort -u | wc -l
make -n train-cv-all 2>&1 | grep -E "train-cv-(clean|neg05|neg10|neg15|neg20)-all"
```
Expected: 第一条输出 `6`(clean 组 for 循环含 6 个配置:3 in_ + 3 cross_test);第二条列出 5 个子目标(train-cv-all 串跑它们)。

- [ ] **Step 3: 确认未破坏 Makefile**

Run: `make -n train-cv-2da >/dev/null && make -n train-clean-all >/dev/null && make help >/dev/null && echo "makefile OK"`
Expected: `makefile OK`(现有目标仍可展开)

- [ ] **Step 4: 提交**

```bash
git add Makefile
git commit -m "build: add train-cv-*-all + train-cv-all targets (30 CV experiments)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## 端到端实跑(需真实数据 + 依赖,人工 checkpoint,非 CI)

> 需 `jianyan` 含 lightgbm/sklearn/pyyaml,且各 `runs/baseline_*/features.csv` 为含 shape-defect 的新 schema(否则先重抽)。

- [ ] 单实验冒烟:`make train-cv-2da PY="conda run -n jianyan python"` → 看 `cv_in_2da_clean.cv.json`(`mode=in_sample`)。
- [ ] 一个 cross_test:`conda run -n jianyan python tools/spec_trainer/src/cv_train.py --config tools/spec_trainer/config/cv_cross_test_2da_clean.yaml --name cv_cross_test_2da_clean --logpath runs/spec_trainer/cv_spec.log` → 看 `cv_cross_test_2da_clean.cv.json`(`mode=cross_test`、`test_per_fold`、`train_oof_auc`)。
- [ ] 全部:`make train-cv-all PY="conda run -n jianyan python"`(30 实验,很慢)。

---

## Self-Review(已核对,记录于此)

- **Spec 覆盖**:§2 cross_test 分支→Task2;§2.1 evaluate_cross_test→Task1;§3 两模式 cv.json 字段→Task2(in_sample/cross_test 各 update);§4 配置生成→Task3;30 配置→Task4;§5 Makefile→Task5;§7 边界(单类 NaN、KeyError 缺列)→Task1/2;§8 测试→各 Task TDD;§9 兼容(不改 main.py/cv_core/既有配置)→全程;cross_test 不做 OOF(§9)→Task2 只 ensemble。✅ 全覆盖。
- **占位符**:无 TBD/TODO;Makefile 5 个 train-cv-*-all 全部展开写出(无"同构"省略);每个代码步给完整代码+命令。✅
- **类型/命名一致**:`evaluate_cross_test(model_paths, X, y) → (ens, per_fold, agg)`(Task1 定义,Task2 调用一致);`to_cv_config(src, name)`(Task3 定义,Task4 经 main 调用);cv 输出 `cv_<name>.cv.json`/`cv_<name>.txt`(Task3 生成 ↔ Task5 `--name $$y` ↔ derive_paths 的 `.json`守卫一致);`mode` ∈ {in_sample, cross_test}。✅
- **依赖隔离**:`evaluate_cross_test` lazy lightgbm;`gen_cv_configs`/其单测无 lightgbm;cross_test 集成测试 `@requires_lgb`。✅

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-cv-matrix-crosstest.md`. Two execution options:

1. **Subagent-Driven(推荐)** — 每 task 派新 subagent + spec/质量双审,task 间 review。
2. **Inline 执行** — 当前会话用 executing-plans 分批 + checkpoint。

Which approach?
