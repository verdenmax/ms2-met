"""Contract test: resolve_feature_cols against the REAL features.csv header.

Existing tests use small synthetic CSVs; this pins the actual stage-2 -> stage-3
column contract (177 cols -> 164 features) using a committed fixture mirroring
runs_new/baseline_2da_clean/features.csv. It would catch schema drift (e.g. the
runs/ 131-col vs runs_new/ 155-col divergence) and accidental label leakage.
"""
import os
import sys

_SPEC_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "spec_trainer", "src")
if _SPEC_SRC not in sys.path:
    sys.path.insert(0, _SPEC_SRC)

from feature_cols import resolve_feature_cols, META_COLUMNS, EXCLUDED_EXTRA  # noqa: E402

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "features_header_177.csv")


def test_real_header_resolves_to_164_features():
    feats = resolve_feature_cols([], [_FIXTURE], "label")
    assert len(feats) == 164


def test_no_label_leakage():
    feats = set(resolve_feature_cols([], [_FIXTURE], "label"))
    assert "label" not in feats
    assert "label_type" not in feats


def test_meta_and_excluded_columns_removed():
    import pandas as pd
    hdr = set(pd.read_csv(_FIXTURE, nrows=0).columns)
    feats = set(resolve_feature_cols([], [_FIXTURE], "label"))
    # every META/EXCLUDED column that is present in the header is dropped
    for c in (META_COLUMNS | EXCLUDED_EXTRA) & hdr:
        assert c not in feats


def test_all_excluded_extra_exist_in_header():
    # guards against dead exclusions silently doing nothing
    import pandas as pd
    hdr = set(pd.read_csv(_FIXTURE, nrows=0).columns)
    assert EXCLUDED_EXTRA <= hdr


def test_heavy_out_of_range_is_retained_as_feature():
    # post-filter it is constant-0, but it is NOT excluded -> still a feature col
    feats = set(resolve_feature_cols([], [_FIXTURE], "label"))
    assert "heavy_out_of_range" in feats
