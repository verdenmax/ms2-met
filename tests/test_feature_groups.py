"""Contract tests for the scientific feature registry."""
import csv
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "tools" / "spec_trainer" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from feature_cols import (  # noqa: E402
    resolve_configured_feature_cols,
    resolve_feature_cols,
)
from feature_groups import (  # noqa: E402
    CONTEXT_FEATURES,
    EVIDENCE_CORE_FEATURES,
    ELIGIBILITY_FEATURES,
    EXPERIMENT_ARMS,
    FEATURE_GROUPS,
    METADATA_COLUMNS,
    MS1_OBSERVED_FEATURES,
    MS2_OBSERVED_FEATURES,
    MS2_PREDICTED_FEATURES,
    TRAINING_EXCLUDED_COLUMNS,
    audit_feature_registry,
    canonical_feature_name,
    feature_spec,
    resolve_experiment_arm,
)


_FIXTURE = _ROOT / "tests" / "fixtures" / "features_header_177.csv"


def _fixture_header() -> list[str]:
    with _FIXTURE.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def test_registry_groups_are_pairwise_disjoint():
    assigned: set[str] = set()
    for name, features in FEATURE_GROUPS.items():
        overlap = assigned.intersection(features)
        assert not overlap, f"group {name} overlaps previous groups: {overlap}"
        assigned.update(features)


def test_real_feature_header_is_exhaustively_registered():
    header = _fixture_header()
    audit = audit_feature_registry(header)

    assert audit.unregistered_columns == ()
    assert audit.ungrouped_model_columns == ()

    auto_features = set(resolve_feature_cols([], [str(_FIXTURE)], "label"))
    registered_model_features = set().union(*FEATURE_GROUPS.values())
    assert auto_features == registered_model_features.intersection(header)


def test_precursor_window_position_is_not_observed_ms1_evidence():
    assert "precursor_centering" in CONTEXT_FEATURES
    assert "precursor_centering" not in MS1_OBSERVED_FEATURES
    assert "precursor_light_centering_defect" in MS1_OBSERVED_FEATURES
    assert "precursor_heavy_centering_defect" in MS1_OBSERVED_FEATURES


def test_evaluability_and_observation_counts_have_correct_groups():
    assert "isotope_model" in METADATA_COLUMNS
    assert "precursor_xic_empty" in ELIGIBILITY_FEATURES
    assert "q1a_total_count" in MS2_OBSERVED_FEATURES
    assert "n_fragments_in_F" in MS2_PREDICTED_FEATURES
    assert "q1a_total_count" not in CONTEXT_FEATURES
    assert "n_fragments_in_F" not in CONTEXT_FEATURES


@pytest.mark.parametrize(
    ("physical", "canonical"),
    [
        ("isotope_correlation", "ms1_heavy_isotope_envelope_cosine"),
        ("mass_shift_error", "ms1_heavy_apex_signed_ppm_error"),
        ("precursor_peak_symmetry", "ms1_heavy_peak_area_asymmetry"),
        ("pred_hl_ratio_cv", "ms2_pred_weighted_log10_hl_std"),
        ("n_fragments_in_F", "ms2_pred_topk_fragment_count"),
        ("all_smoothness_mean", "all_roughness_mean"),
    ],
)
def test_legacy_columns_have_stable_canonical_names(physical, canonical):
    assert canonical_feature_name(physical) == canonical
    assert feature_spec(physical).canonical_name == canonical


def test_canonical_names_are_unique_except_deprecated_shift_alias():
    reverse = {}
    for column in _fixture_header():
        canonical = canonical_feature_name(column)
        reverse.setdefault(canonical, []).append(column)
    duplicates = {name: columns for name, columns in reverse.items()
                  if len(columns) > 1}
    assert duplicates == {}
    assert canonical_feature_name("total_silac_shift") == (
        canonical_feature_name("total_label_shift"))


@pytest.mark.parametrize("arm", sorted(EXPERIMENT_ARMS))
def test_formal_arms_exclude_metadata_training_exclusions_and_flags(arm):
    features = set(resolve_experiment_arm(arm, _fixture_header()))

    assert features.isdisjoint(METADATA_COLUMNS)
    assert features.isdisjoint(TRAINING_EXCLUDED_COLUMNS)
    assert features.isdisjoint(ELIGIBILITY_FEATURES)


def test_evidence_core_is_compact_paired_evidence():
    features = set(resolve_experiment_arm("evidence_core", _fixture_header()))
    assert features == set(EVIDENCE_CORE_FEATURES)
    assert len(features) == 35
    assert {"precursor_pearson", "all_p75", "pred_coverage_wpred"} <= features
    assert features.isdisjoint({
        "precursor_light_max_int", "b_count", "y_count", "all_count",
        "q1a_total_count", "n_fragments_in_F",
    })


def test_resolve_arm_preserves_header_order_and_prefers_canonical_shift():
    header = [
        "label", "total_silac_shift", "precursor_centering",
        "total_label_shift", "kr_count",
    ]
    assert resolve_experiment_arm("context_only", header) == [
        "precursor_centering", "total_label_shift", "kr_count",
    ]


def test_strict_arm_resolution_rejects_schema_drift():
    with pytest.raises(ValueError, match="mystery_feature"):
        resolve_experiment_arm(
            "ms1_only", ["precursor_pearson", "mystery_feature"])


def test_configured_arm_resolves_registry_and_applies_drop_list():
    data_cfg = {
        "feature_arm": "ms2_all",
        "feature_cols": [],
        "drop_features": ["spec_pattern_spearman_b", "spec_pattern_SA_b"],
    }
    features = resolve_configured_feature_cols(
        data_cfg, [str(_FIXTURE)], "label")

    assert "all_count" in features
    assert "spec_pattern_SA" in features
    assert "spec_pattern_spearman_b" not in features
    assert "spec_pattern_SA_b" not in features
    assert set(features).isdisjoint(ELIGIBILITY_FEATURES)


def test_configured_arm_rejects_explicit_feature_cols():
    with pytest.raises(ValueError, match="feature_arm.*feature_cols"):
        resolve_configured_feature_cols(
            {"feature_arm": "ms1_only",
             "feature_cols": ["precursor_pearson"]},
            [str(_FIXTURE)], "label")


def test_configured_arm_rejects_unknown_drop_feature():
    with pytest.raises(ValueError, match="unknown drop_features"):
        resolve_configured_feature_cols(
            {"feature_arm": "ms1_only",
             "drop_features": ["typo_feature"]},
            [str(_FIXTURE)], "label")


def test_complete_arm_rejects_missing_registered_feature(tmp_path):
    path = tmp_path / "partial.csv"
    path.write_text("sequence,label,precursor_pearson\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema drift.*missing"):
        resolve_configured_feature_cols(
            {"feature_arm": "ms1_only", "require_complete_arm": True},
            [str(path)], "label")
