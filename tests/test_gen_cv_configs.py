import copy
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "spec_trainer"))

from src.feature_groups import FORMAL_DROP_FEATURES


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
    assert cv["training"]["min_class_groups_per_split"] == 5
    assert cv["operating_point"] == {
        "target_fprs": [0.05, 0.10], "primary_target_fpr": 0.10}
    assert cv["evaluation_semantics"]["positive_class"] == \
        "incorrect_identification"
    assert cv["data"]["feature_arm"] == "evidence_all"
    assert cv["data"]["cohort"] == "evidence_common"
    assert cv["data"]["drop_features"] == list(FORMAL_DROP_FEATURES)
    assert cv["data"]["feature_cols"] == []
    assert "test_size" not in cv["data"]
    assert cv["data"]["require_complete_arm"] is True
    assert cv["audit"] == {"suspect_threshold": 0.9, "suspect_top_n": 200}
    assert cv["output"]["model_path"] == "runs/spec_trainer/models/cv_in_2da_clean.txt"
    assert cv["output"]["result_path"] == "runs/spec_trainer/results/cv_in_2da_clean.cv.json"
    assert "figures_dir" not in cv["output"]
    assert cv["data"]["train_files"] == _src_in()["data"]["train_files"]   # 保留
    assert cv["training"]["num_boost_round"] == 2000
    assert cv["training"]["early_stopping_first_metric_only"] is True
    assert cv["model"]["params"]["bagging_freq"] == 1
    assert cv["model"]["params"]["deterministic"] is True
    assert "is_unbalance" not in cv["model"]["params"]
    assert "scale_pos_weight" not in cv["model"]["params"]


def test_to_cv_config_supports_compact_evidence_core():
    from gen_cv_configs import to_cv_config
    cv = to_cv_config(_src_in(), "in_2da_clean", feature_arm="evidence_core")
    assert cv["data"]["feature_arm"] == "evidence_core"


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


def test_to_cv_config_rebases_inputs_and_outputs():
    from gen_cv_configs import to_cv_config
    cv = to_cv_config(
        _src_in(), "in_2da_clean",
        feature_root="/data/features-v2",
        output_root="/results/cv-v2",
    )
    assert cv["data"]["train_files"] == [
        "/data/features-v2/baseline_2da_clean/features.csv"]
    assert cv["output"]["model_path"] == \
        "/results/cv-v2/models/cv_in_2da_clean.txt"
    assert cv["output"]["result_path"] == \
        "/results/cv-v2/results/cv_in_2da_clean.cv.json"


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


def test_make_train_cv_all_uses_external_roots(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    # Make prerequisite paths intentionally require a whitespace-free root;
    # GNU make tokenizes prerequisite lists on spaces.
    feature_root = tmp_path / "feature_snapshot"
    output_root = tmp_path / "cv_output"
    for fdr in ("clean", "neg05", "neg10", "neg15", "neg20"):
        for dataset in ("2da", "5da", "normal"):
            path = feature_root / f"baseline_{dataset}_{fdr}" / "features.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    result = subprocess.run(
        ["make", "-n", "train-cv-all",
         f"FEATURE_ROOT={feature_root}",
         f"CV_OUTPUT_ROOT={output_root}"],
        cwd=project_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert f'--feature-root "{feature_root}"' in output
    assert f'--output-root "{output_root}"' in output
    assert f'--config-dir "{output_root}/configs"' in output
    assert '--feature-arm "evidence_all"' in output
    assert f'--config "{output_root}/configs/$y.yaml"' in output
    assert f'--logpath "{output_root}/logs/$y.log"' in output
    assert "--config tools/spec_trainer/config/$y.yaml" not in output


def test_committed_cv_matrix_uses_formal_evidence_contract():
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    paths = sorted((project_root / "tools/spec_trainer/config").glob(
        "cv_*.yaml"))
    assert len(paths) == 30
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        data = cfg["data"]
        assert data["feature_arm"] == "evidence_all", path.name
        assert data["cohort"] == "evidence_common", path.name
        assert data["drop_features"] == list(FORMAL_DROP_FEATURES), path.name
