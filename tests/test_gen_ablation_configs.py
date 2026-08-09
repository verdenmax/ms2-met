import sys
import subprocess
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "spec_trainer"
if str(_TOOL) not in sys.path:
    sys.path.insert(0, str(_TOOL))

from gen_ablation_configs import (  # noqa: E402
    ABLATION_ARMS,
    build_ablation_config,
    generate_configs,
)


def test_build_ablation_config_uses_external_root_and_common_cohort(tmp_path):
    cfg = build_ablation_config(
        feature_root=tmp_path / "features",
        output_root=tmp_path / "out",
        dataset="2da",
        fdr="neg20",
        arm="ms1_only",
    )

    data = cfg["data"]
    assert data["train_files"] == [str(
        tmp_path / "features" / "baseline_2da_neg20" / "features.csv")]
    assert data["feature_arm"] == "ms1_only"
    assert data["cohort"] == "evidence_common"
    assert data["drop_features"] == [
        "spec_pattern_spearman_b", "spec_pattern_SA_b"]
    assert data["group_col"] == "sequence"
    assert cfg["training"]["cv_folds"] == 5


def test_generate_configs_writes_one_yaml_per_dataset_and_arm(tmp_path):
    feature_root = tmp_path / "features"
    for dataset in ("2da", "5da"):
        feature_dir = feature_root / f"baseline_{dataset}_neg20"
        feature_dir.mkdir(parents=True)
        (feature_dir / "features.csv").write_text("label\n", encoding="utf-8")

    paths = generate_configs(
        feature_root=feature_root,
        output_root=tmp_path / "out",
        config_dir=tmp_path / "cfg",
        datasets=["2da", "5da"],
        arms=list(ABLATION_ARMS),
        fdr="neg20",
    )

    assert len(paths) == 2 * len(ABLATION_ARMS)
    assert all(path.exists() for path in paths)


def test_generate_configs_fails_fast_when_feature_csv_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="baseline_2da_neg20"):
        generate_configs(
            feature_root=tmp_path / "features",
            output_root=tmp_path / "out",
            config_dir=tmp_path / "cfg",
            datasets=["2da"],
            arms=["ms1_only"],
            fdr="neg20",
        )


@pytest.mark.parametrize(
    "target", ["train-ablation-neg20-2da", "train-ablation-neg20"])
def test_ablation_make_targets_are_dry_runnable(target):
    proc = subprocess.run(
        ["make", "-n", target, "FEATURE_ROOT=/tmp/external-features"],
        cwd=_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "gen_ablation_configs.py" in proc.stdout
    assert "cv_train.py" in proc.stdout
    assert "/tmp/external-features" in proc.stdout
