"""Adapter from the existing spec_trainer protocol to neural experiments.

This is the only deep_trainer module allowed to know fixed_negpool's internal
column names.  Neural encoders consume ``PreparedProtocol`` and therefore
cannot accidentally drift from the frozen cohort/split/fold contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPEC_SRC = _PROJECT_ROOT / "tools" / "spec_trainer" / "src"
if str(_SPEC_SRC) not in sys.path:
    sys.path.insert(0, str(_SPEC_SRC))

from fixed_negpool import (  # noqa: E402
    MODEL_TIERS,
    _DATASET,
    _OUTER_FOLD,
    _SAMPLE_ID,
    _SOURCE_ROW,
    _SPLIT,
    _TIER,
    _assert_formal_config,
    feature_paths,
    prepare_combined_fixed_negpool,
    prepare_fixed_negpool,
)
from cv_train import _operating_targets, _validate_frame  # noqa: E402


@dataclass
class PreparedProtocol:
    frame: pd.DataFrame
    membership: pd.DataFrame
    group_fold_map: pd.DataFrame
    feature_cols: list[str]
    identity_cols: list[str]
    validation: dict
    feature_arm: str
    cohort: str
    target_col: str
    group_col: str
    sample_id_col: str
    source_row_col: str
    dataset_col: str
    tier_col: str
    split_col: str
    outer_fold_col: str
    inner_valid_cols: list[str]
    model_tiers: dict[str, tuple[str, ...]]
    target_fprs: list[float]

    def training_frame(self, model_name: str) -> pd.DataFrame:
        if model_name not in self.model_tiers:
            raise ValueError(
                f"unknown model pool {model_name!r}; expected "
                f"{sorted(self.model_tiers)}")
        train = self.frame[self.frame[self.split_col].eq("train")]
        allowed = self.model_tiers[model_name]
        use = train[self.target_col].eq(1) | train[self.tier_col].isin(allowed)
        result = train.loc[use].copy().reset_index(drop=True)
        _validate_frame(
            result, self.feature_cols, self.target_col, self.group_col)
        return result

    def test_frame(self) -> pd.DataFrame:
        result = self.frame[
            self.frame[self.split_col].eq("test")].copy().reset_index(drop=True)
        _validate_frame(
            result, self.feature_cols, self.target_col, self.group_col)
        return result


def prepare_protocol(
    split_config_path: str,
    feature_root: str,
    dataset: str,
    *,
    test_fraction: float = 0.20,
    min_test_errors_per_tier: int = 100,
    split_candidates: int = 128,
) -> PreparedProtocol:
    """Prepare the exact fixed-negpool protocol used by the tree baseline."""
    with open(split_config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _assert_formal_config(config)
    if dataset == "combined":
        prepared = prepare_combined_fixed_negpool(
            feature_root,
            config,
            test_fraction=test_fraction,
            min_test_errors_per_tier=min_test_errors_per_tier,
            split_candidates=split_candidates,
        )
    elif dataset in {"2da", "5da", "normal"}:
        prepared = prepare_fixed_negpool(
            feature_paths(feature_root, dataset),
            config,
            test_fraction=test_fraction,
            min_test_errors_per_tier=min_test_errors_per_tier,
            split_candidates=split_candidates,
        )
    else:
        raise ValueError("dataset must be combined, 2da, 5da, or normal")

    n_folds = int(config["training"].get("cv_folds", 5))
    targets, _ = _operating_targets(config)
    return PreparedProtocol(
        frame=prepared.frame,
        membership=prepared.membership,
        group_fold_map=prepared.group_fold_map,
        feature_cols=prepared.feature_cols,
        identity_cols=prepared.identity_cols,
        validation=prepared.validation,
        feature_arm=config["data"]["feature_arm"],
        cohort=config["data"]["cohort"],
        target_col=config["data"]["target_col"],
        group_col=config["data"]["group_col"],
        sample_id_col=_SAMPLE_ID,
        source_row_col=_SOURCE_ROW,
        dataset_col=_DATASET,
        tier_col=_TIER,
        split_col=_SPLIT,
        outer_fold_col=_OUTER_FOLD,
        inner_valid_cols=[
            f"inner_valid_for_fold_{fold}" for fold in range(n_folds)],
        model_tiers={key: tuple(value) for key, value in MODEL_TIERS.items()},
        target_fprs=list(targets),
    )
