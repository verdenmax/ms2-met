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


def test_derive_paths_rejects_colliding_result_path():
    import cv_train, pytest
    cfg = {"output": {"model_path": "runs/m/x.txt",
                      "result_path": "runs/r/x.out"}}   # not .json -> would collide
    with pytest.raises(ValueError):
        cv_train.derive_paths(cfg)


def test_read_dataframe_concat(tmp_path):
    import cv_train
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(a, index=False)
    pd.DataFrame({"x": [3]}).to_csv(b, index=False)
    df = cv_train.read_dataframe([str(a), str(b)])
    assert list(df["x"]) == [1, 2, 3]


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


def test_inner_split_no_leak_grouped():
    import cv_train
    # te_idx occupies LOW indices, tr_idx is OFFSET high -> the local->global
    # remap is NON-trivial: a dropped remap would return low (0..) indices that
    # overlap te_idx and fail the disjointness assertion (gives the test teeth).
    n = 60
    X = pd.DataFrame({"f": np.arange(n)})
    y = pd.Series([1, 0] * (n // 2))                  # both classes everywhere
    groups = pd.Series(np.repeat(np.arange(n // 2), 2))   # 30 peptides x2 rows
    te_idx = np.arange(0, 12)                         # held-out OOF fold (low)
    tr_idx = np.arange(12, n)                         # train fold (offset high)
    tr2, val = cv_train._inner_split(X, y, groups, tr_idx, valid_size=0.25, seed=42)
    assert set(val).issubset(set(tr_idx))             # val drawn only from train
    assert set(tr2).issubset(set(tr_idx))
    assert set(val).isdisjoint(set(te_idx))           # ** no leak into OOF fold **
    assert set(tr2).isdisjoint(set(te_idx))
    assert set(tr2).isdisjoint(set(val))              # train/val partition
    assert len(tr2) + len(val) == len(tr_idx)
    assert set(groups.iloc[tr2]).isdisjoint(set(groups.iloc[val]))   # no peptide spans


def test_inner_split_no_groups_branch():
    import cv_train
    n = 50
    X = pd.DataFrame({"f": np.arange(n)})
    y = pd.Series([1, 0] * (n // 2))                  # both classes for stratify
    te_idx = np.arange(0, 10)                         # low
    tr_idx = np.arange(10, n)                         # offset high
    tr2, val = cv_train._inner_split(X, y, None, tr_idx, valid_size=0.25, seed=1)
    assert set(val).issubset(set(tr_idx))
    assert set(val).isdisjoint(set(te_idx))           # ** no leak **
    assert set(tr2).isdisjoint(set(te_idx))
    assert set(tr2).isdisjoint(set(val))
    assert len(tr2) + len(val) == len(tr_idx)


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
    assert res["name"] == "toy"


@requires_lgb
def test_predict_ensemble_in_range(tmp_path):
    import cv_train, lightgbm as lgb
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
    # teeth: it is the MEAN of the per-fold predictions, not fold-0 / max / min
    per_fold = [lgb.Booster(model_file=p).predict(X.values) for p in model_paths]
    assert np.allclose(s, np.mean(per_fold, axis=0))
    # DataFrame input path (docstring promises numpy OR DataFrame) agrees with numpy
    s_df = cv_train.predict_ensemble(model_paths, X)
    assert np.allclose(s_df, s)
