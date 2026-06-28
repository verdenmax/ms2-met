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


def test_to_cv_config_rejects_bad_shape():
    import pytest
    from gen_cv_configs import to_cv_config
    with pytest.raises(ValueError):
        to_cv_config(None, "x")                      # not a dict
    with pytest.raises(ValueError):
        to_cv_config({"data": {}}, "x")              # missing 'training'
