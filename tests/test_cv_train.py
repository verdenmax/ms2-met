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
