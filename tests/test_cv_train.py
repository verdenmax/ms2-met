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
    assert list(df["__source_row"]) == [0, 1, 0]
    assert set(df["__source_file"]) == {str(a.resolve()), str(b.resolve())}


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
            "metric": ["auc", "binary_logloss"],
            "bagging_fraction": 0.8, "bagging_freq": 1,
            "seed": 42, "feature_fraction_seed": 42, "bagging_seed": 42,
            "data_random_seed": 42, "deterministic": True,
            "force_col_wise": True, "min_data_in_leaf": 5, "verbose": -1}},
        "training": {"num_boost_round": 40, "early_stopping_rounds": 15,
                     "cv_folds": 5, "cv_seed": 42, "valid_size": 0.25,
                     "min_class_groups_per_split": 2,
                     "early_stopping_first_metric_only": True},
        "operating_point": {"target_fpr": 0.10},
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
    assert len(fold_metrics) == 5 and "roc_auc" in fold_metrics[0]
    counts = fold_metrics[0]["split_counts"]
    assert set(counts) == {"train", "valid", "oof_test"}
    for split in counts.values():
        assert split["n_correct_groups"] >= 2
        assert split["n_error_groups"] >= 2
        assert split["n_mixed_groups"] == 0
    assert len(model_paths) == 5 and all(os.path.exists(p) for p in model_paths)


def test_inner_split_no_leak_grouped():
    import cv_train
    # te_idx occupies LOW indices, tr_idx is OFFSET high -> the local->global
    # remap is NON-trivial: a dropped remap would return low (0..) indices that
    # overlap te_idx and fail the disjointness assertion (gives the test teeth).
    n = 60
    X = pd.DataFrame({"f": np.arange(n)})
    groups = pd.Series(np.repeat(np.arange(n // 2), 2))   # 30 peptides x2 rows
    y = pd.Series((groups.to_numpy() % 2).astype(int))  # one label per peptide
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
    assert set(y.iloc[tr2]) == {0, 1}
    assert set(y.iloc[val]) == {0, 1}


def test_inner_split_supports_parent_positive_with_synthetic_negative_group():
    import cv_train
    X = pd.DataFrame({"f": np.arange(12)})
    groups = pd.Series(np.repeat(np.arange(6), 2))
    y = pd.Series([0, 1] * 6)  # synthetic negative + parent positive
    tr, val = cv_train._inner_split(
        X, y, groups, np.arange(12), valid_size=0.34, seed=42)
    assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[val]))
    assert set(y.iloc[tr]) == {0, 1}
    assert set(y.iloc[val]) == {0, 1}


def test_validate_split_counts_rejects_too_few_minority_groups():
    import cv_train
    counts = {"valid": {
        "n_rows": 10, "n_correct": 9, "n_error": 1, "n_groups": 10,
        "n_correct_groups": 9, "n_error_groups": 1,
        "n_mixed_groups": 0,
    }}
    with pytest.raises(ValueError, match="require >= 2"):
        cv_train._validate_split_counts(
            fold=0, split_counts=counts, min_class_groups=2, grouped=True)


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


def test_predefined_splits_preserve_group_assignments():
    import cv_train
    groups = pd.Series(["a", "a", "b", "b", "c", "c", "d", "d"])
    fold_ids = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    splits = cv_train._predefined_cv_splits(
        fold_ids, len(fold_ids), 2, groups=groups)
    assert [test.tolist() for _, test in splits] == [
        [0, 1, 4, 5], [2, 3, 6, 7]]
    with pytest.raises(ValueError, match="split at least one group"):
        cv_train._predefined_cv_splits(
            np.array([0, 1, 1, 1, 0, 0, 1, 1]), 8, 2,
            groups=groups)


def test_predefined_inner_split_rejects_outer_overlap():
    import cv_train
    groups = pd.Series(["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="outer OOF"):
        cv_train._predefined_inner_split(
            np.array([0, 1]), np.array([2, 3]),
            np.array([False, True, True, False]), 4, groups=groups)


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
    assert ("roc_auc" in res and "error_pr_auc" in res
            and "fnr_at_fpr5" in res)
    assert len(res["fold_metrics"]) == 5 and "roc_auc_mean" in res
    assert "error_pr_auc_mean" in res and "error_pr_auc_std" in res
    assert "fnr_at_fpr5_mean" in res and "fnr_at_fpr5_std" in res
    assert res["operating_point"]["target_fpr"] == 0.10
    assert res["operating_point"]["threshold_source"] == \
        "pooled_train_oof_single_member_scores"
    assert res["operating_point"]["positive_class"] == \
        "incorrect_identification"
    assert res["operating_point"]["train_oof_metrics"]["fpr"] <= 0.10
    assert (tmp_path / "r.cv.suspects.csv").exists()     # 派生路径
    assert (tmp_path / "r.cv.oof.csv").exists()
    assert not (tmp_path / "r.cv.test_scores.csv").exists()
    assert res["provenance"]["git_commit"]
    assert res["fold_dispersion_note"].endswith("confidence interval")
    assert all(m["best_iteration"] >= 1 for m in res["fold_metrics"])
    assert summary["roc_auc"] == res["roc_auc"]
    assert res["name"] == "toy"


@requires_lgb
def test_main_applies_feature_arm_and_common_cohort(tmp_path):
    import cv_train, yaml
    df = _toy_df()
    for column, value in {
        "heavy_in_raw": 1,
        "heavy_out_of_range": 0,
        "precursor_xic_empty": 0,
        "q1a_valid": 1,
        "has_lib_pred": 1,
        "isotope_model_valid": 1,
    }.items():
        df[column] = value
    df.loc[0, "has_lib_pred"] = 0
    csv = tmp_path / "feat.csv"
    df.to_csv(csv, index=False)
    cfg = _toy_cfg(tmp_path)
    cfg["data"].update({
        "train_files": [str(csv)],
        "feature_arm": "ms1_only",
        "cohort": "evidence_common",
        "drop_features": ["spec_pattern_spearman_b", "spec_pattern_SA_b"],
    })
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    cv_train.main(["--config", str(cfg_path), "--name", "arm",
                   "--logpath", str(tmp_path / "log.txt")])

    result = json.loads((tmp_path / "r.cv.json").read_text())
    experiment = result["experiment"]
    assert experiment["feature_arm"] == "ms1_only"
    assert experiment["cohort"] == "evidence_common"
    assert experiment["feature_cols"] == ["precursor_pearson"]
    assert experiment["n_features"] == 1
    assert experiment["train_cohort"]["before"]["n_rows"] == len(df)
    assert experiment["train_cohort"]["after"]["n_rows"] == len(df) - 1


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


@requires_lgb
def test_evaluate_cross_test(tmp_path):
    import cv_train, lightgbm as lgb
    dfA = _toy_df(seed=0)               # 训练数据集 A
    dfB = _toy_df(seed=7)               # 外部测试数据集 B（不同抽样）
    csvA = tmp_path / "a.csv"; dfA.to_csv(csvA, index=False)
    feature_cols = resolve_feature_cols(None, [str(csvA)], "label")
    cfg = _toy_cfg(tmp_path)
    _, fold_metrics, model_paths = cv_train.assemble_oof(
        dfA, dfA[feature_cols], dfA["label"], dfA["sequence"],
        cfg, feature_cols, str(tmp_path / "m"))
    Xb = dfB[feature_cols].values
    yb = dfB["label"].values
    ens, per_fold, agg = cv_train.evaluate_cross_test(
        model_paths, dfB[feature_cols], yb, fold_metrics=fold_metrics)
    assert ens.shape == (len(dfB),)
    # ensemble = 各折预测的均值
    per = [lgb.Booster(model_file=p).predict(Xb) for p in model_paths]
    assert np.allclose(ens, np.mean(per, axis=0))
    assert (len(per_fold) == 5 and "roc_auc" in per_fold[0]
            and "error_pr_auc" in per_fold[0]
            and "oracle_test_fnr_at_fpr5" in per_fold[0])
    assert {"member_model_roc_auc_mean", "member_model_roc_auc_std",
            "member_model_error_pr_auc_mean",
            "member_model_error_pr_auc_std",
            "member_model_oracle_fnr_at_fpr5_mean",
            "member_model_oracle_fnr_at_fpr5_std"} <= set(agg)
    assert "fpr_10" in agg["locked_operating_points"]
    locked = agg["locked_operating_points"]["fpr_10"]
    assert locked["method"] == "fold_calibrated_majority_vote"
    assert len(locked["member_error_thresholds"]) == 5
    # value-level: per-fold metrics are the real auc/fnr (not swapped)
    from cv_core import fnr_at_fpr5 as _fnr
    from sklearn.metrics import roc_auc_score as _auc
    assert np.isclose(per_fold[0]["roc_auc"], _auc(yb, per[0]))
    assert np.isclose(
        per_fold[0]["oracle_test_fnr_at_fpr5"], _fnr(yb, per[0]))
    assert np.isclose(agg["member_model_roc_auc_mean"],
                      np.mean([m["roc_auc"] for m in per_fold]))
    # single-class external y -> per-fold + agg NaN, but ensemble still scores
    import numpy as _np
    ens1, pf1, agg1 = cv_train.evaluate_cross_test(
        model_paths, dfB[feature_cols], _np.ones(len(dfB)),
        fold_metrics=fold_metrics)
    assert (_np.isnan(pf1[0]["roc_auc"])
            and _np.isnan(agg1["member_model_roc_auc_mean"]))
    assert _np.isfinite(ens1).all()

    with pytest.raises(ValueError, match="feature schema mismatch"):
        cv_train.evaluate_cross_test(
            model_paths, dfB[list(reversed(feature_cols))], yb)


@requires_lgb
def test_main_cross_test_mode(tmp_path):
    import cv_train, yaml
    dfA = _toy_df(seed=0)                # 训练数据集 A
    dfB = _toy_df(n_groups=30, seed=7)  # 外部测试集 B：150 行（≠ A 的 200），规模不同才能区分评估目标
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
    assert len(res["member_model_retrospective_metrics"]) == 5
    assert "member_model_roc_auc_mean" in res and "train_oof_roc_auc" in res
    assert "fnr_at_fpr5" not in res
    assert res["retrospective_test_working_points"][
        "threshold_source"] == "external_test_labels"
    op = res["operating_point"]
    assert op["target_fpr"] == 0.10
    assert op["threshold_source"] == \
        "pooled_train_oof_single_member_scores"
    assert "test_metrics" in op
    assert op["external_ensemble"]["method"] == \
        "fold_calibrated_majority_vote"
    assert op["test_metrics"]["n_actual_correct"] == int(
        (dfB["label"] == 1).sum())
    assert res["n_actual_correct"] == int((dfB["label"] == 1).sum())
    assert res["n_actual_error"] == int((dfB["label"] == 0).sum())
    assert res["n_actual_correct"] + res["n_actual_error"] == len(dfB)
    assert (tmp_path / "r.cv.suspects.csv").exists()
    assert (tmp_path / "r.cv.oof.csv").exists()
    assert (tmp_path / "r.cv.test_scores.csv").exists()
    assert res["experiment"]["test_missingness"]["by_class"]
    assert res["experiment"]["train_test_sequence_overlap"][
        "test_overlap_fraction"] == 1.0
    assert summary["roc_auc"] == res["roc_auc"]


def test_main_fails_closed_when_group_column_is_missing(tmp_path):
    import cv_train, yaml
    df = _toy_df().drop(columns="sequence")
    csv = tmp_path / "feat.csv"
    df.to_csv(csv, index=False)
    cfg = _toy_cfg(tmp_path)
    cfg["data"]["train_files"] = [str(csv)]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="refusing.*ungrouped CV"):
        cv_train.main(["--config", str(cfg_path), "--name", "missing-group",
                       "--logpath", str(tmp_path / "log.txt")])


def test_validate_frame_rejects_bad_labels_and_infinite_features():
    import cv_train
    bad_label = pd.DataFrame({
        "sequence": ["A", "B"], "label": [1, 2], "f": [0.1, 0.2]})
    with pytest.raises(ValueError, match="only 0.*and 1"):
        cv_train._validate_frame(
            bad_label, ["f"], "label", group_col="sequence")

    bad_feature = pd.DataFrame({
        "sequence": ["A", "B"], "label": [1, 0], "f": [0.1, np.inf]})
    with pytest.raises(ValueError, match="infinite values"):
        cv_train._validate_frame(
            bad_feature, ["f"], "label", group_col="sequence")


@requires_lgb
def test_main_refuses_to_overwrite_completed_bundle(tmp_path):
    import cv_train, yaml
    df = _toy_df()
    csv = tmp_path / "feat.csv"
    df.to_csv(csv, index=False)
    cfg = _toy_cfg(tmp_path)
    cfg["data"]["train_files"] = [str(csv)]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    argv = ["--config", str(cfg_path), "--name", "once",
            "--logpath", str(tmp_path / "log.txt")]
    cv_train.main(argv)
    with pytest.raises(FileExistsError, match="--overwrite"):
        cv_train.main(argv)
