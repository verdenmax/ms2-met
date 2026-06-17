"""Tests for workflows.feature_postfilter.filter_heavy_out_of_range.

Policy (locked here): a PSM whose heavy SILAC precursor fell outside the
acquisition m/z range (heavy_out_of_range == 1) lacks the heavy channel and
cannot be validated by fragment evidence. Such rows are dropped for BOTH
classes (positive AND negative) — symmetric, to avoid teaching the model a
spurious "out-of-range => positive" rule.
"""
import pandas as pd

from workflows.feature_postfilter import filter_heavy_out_of_range


def _df(rows):
    return pd.DataFrame(rows)


def test_drops_both_classes_out_of_range():
    df = _df([
        {"sequence": "AK", "label": 1, "label_type": "positive", "heavy_out_of_range": 0},
        {"sequence": "BK", "label": 1, "label_type": "positive", "heavy_out_of_range": 1},
        {"sequence": "CK", "label": 0, "label_type": "negative", "heavy_out_of_range": 0},
        {"sequence": "DK", "label": 0, "label_type": "negative", "heavy_out_of_range": 1},
    ])
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    assert list(kept["sequence"]) == ["AK", "CK"]
    assert n_pos == 1
    assert n_neg == 1


def test_keeps_all_when_none_out_of_range():
    df = _df([
        {"sequence": "AK", "label": 1, "label_type": "positive", "heavy_out_of_range": 0},
        {"sequence": "CK", "label": 0, "label_type": "negative", "heavy_out_of_range": 0},
    ])
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    assert len(kept) == 2
    assert (n_pos, n_neg) == (0, 0)


def test_missing_column_is_noop():
    df = _df([
        {"sequence": "AK", "label": 1, "label_type": "positive"},
        {"sequence": "CK", "label": 0, "label_type": "negative"},
    ])
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    assert len(kept) == 2
    assert (n_pos, n_neg) == (0, 0)


def test_empty_dataframe():
    df = _df([]).reindex(columns=["sequence", "label", "label_type", "heavy_out_of_range"])
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    assert len(kept) == 0
    assert (n_pos, n_neg) == (0, 0)


def test_resets_index_after_drop():
    df = _df([
        {"sequence": "AK", "label": 1, "label_type": "positive", "heavy_out_of_range": 1},
        {"sequence": "BK", "label": 1, "label_type": "positive", "heavy_out_of_range": 0},
    ])
    kept, _, _ = filter_heavy_out_of_range(df)
    assert list(kept.index) == [0]


def test_robust_to_string_and_bool_values():
    df = _df([
        {"sequence": "AK", "label": 1, "label_type": "positive", "heavy_out_of_range": "1"},
        {"sequence": "BK", "label": 0, "label_type": "negative", "heavy_out_of_range": True},
        {"sequence": "CK", "label": 1, "label_type": "positive", "heavy_out_of_range": "0"},
        {"sequence": "DK", "label": 0, "label_type": "negative", "heavy_out_of_range": False},
    ])
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    assert list(kept["sequence"]) == ["CK", "DK"]
    assert (n_pos, n_neg) == (1, 1)


def test_counts_fall_back_to_label_when_no_label_type():
    df = _df([
        {"sequence": "AK", "label": 1, "heavy_out_of_range": 1},
        {"sequence": "DK", "label": 0, "heavy_out_of_range": 1},
        {"sequence": "CK", "label": 1, "heavy_out_of_range": 0},
    ])
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    assert list(kept["sequence"]) == ["CK"]
    assert (n_pos, n_neg) == (1, 1)


# --- CLI / file-level helper ---

from workflows.feature_postfilter import filter_csv_file  # noqa: E402

_ROWS = [
    {"sequence": "AK", "label": 1, "label_type": "positive", "heavy_out_of_range": 0},
    {"sequence": "BK", "label": 1, "label_type": "positive", "heavy_out_of_range": 1},
    {"sequence": "DK", "label": 0, "label_type": "negative", "heavy_out_of_range": 1},
]


def test_filter_csv_dry_run_does_not_write(tmp_path):
    import pandas as pd
    p = tmp_path / "features.csv"
    pd.DataFrame(_ROWS).to_csv(p, index=False)
    n_pos, n_neg = filter_csv_file(str(p))  # dry-run
    assert (n_pos, n_neg) == (1, 1)
    assert len(pd.read_csv(p)) == 3  # unchanged


def test_filter_csv_output_writes_filtered(tmp_path):
    import pandas as pd
    p = tmp_path / "features.csv"
    out = tmp_path / "features.filtered.csv"
    pd.DataFrame(_ROWS).to_csv(p, index=False)
    filter_csv_file(str(p), output=str(out))
    assert len(pd.read_csv(p)) == 3            # input untouched
    res = pd.read_csv(out)
    assert list(res["sequence"]) == ["AK"]     # both out-of-range rows dropped


def test_filter_csv_in_place_with_backup(tmp_path):
    import pandas as pd
    p = tmp_path / "features.csv"
    pd.DataFrame(_ROWS).to_csv(p, index=False)
    filter_csv_file(str(p), in_place=True)
    assert list(pd.read_csv(p)["sequence"]) == ["AK"]      # overwritten/filtered
    assert (tmp_path / "features.csv.prefilter.bak").exists()
    assert len(pd.read_csv(tmp_path / "features.csv.prefilter.bak")) == 3


# --- NaN-coerce + no-class-column branches ---

def test_nan_or_unparseable_heavy_out_of_range_is_kept():
    df = _df([
        {"sequence": "AK", "label": 1, "label_type": "positive", "heavy_out_of_range": float("nan")},
        {"sequence": "BK", "label": 0, "label_type": "negative", "heavy_out_of_range": "weird"},
        {"sequence": "CK", "label": 0, "label_type": "negative", "heavy_out_of_range": 1},
    ])
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    # NaN / non-numeric coerce to NaN -> not ==1 -> kept; only the real 1 drops
    assert list(kept["sequence"]) == ["AK", "BK"]
    assert (n_pos, n_neg) == (0, 1)


def test_counts_without_label_or_label_type():
    df = _df([
        {"sequence": "AK", "heavy_out_of_range": 1},
        {"sequence": "BK", "heavy_out_of_range": 1},
        {"sequence": "CK", "heavy_out_of_range": 0},
    ])
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    assert list(kept["sequence"]) == ["CK"]
    # no label/label_type -> all drops counted as n_pos, n_neg=0 (line 52-54)
    assert (n_pos, n_neg) == (2, 0)


# --- CLI main() ---

from workflows.feature_postfilter import main as cli_main  # noqa: E402
import pytest  # noqa: E402


def test_cli_output_with_multiple_inputs_errors(tmp_path):
    import pandas as pd
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    pd.DataFrame(_ROWS).to_csv(a, index=False)
    pd.DataFrame(_ROWS).to_csv(b, index=False)
    with pytest.raises(SystemExit):
        cli_main([str(a), str(b), "--output", str(tmp_path / "o.csv")])


def test_cli_output_with_in_place_errors(tmp_path):
    import pandas as pd
    a = tmp_path / "a.csv"
    pd.DataFrame(_ROWS).to_csv(a, index=False)
    with pytest.raises(SystemExit):
        cli_main([str(a), "--output", str(tmp_path / "o.csv"), "--in-place"])


def test_cli_no_positional_errors():
    with pytest.raises(SystemExit):
        cli_main([])


def test_cli_in_place_no_backup(tmp_path):
    import pandas as pd
    a = tmp_path / "a.csv"
    pd.DataFrame(_ROWS).to_csv(a, index=False)
    cli_main([str(a), "--in-place", "--no-backup"])
    assert list(pd.read_csv(a)["sequence"]) == ["AK"]   # filtered in place
    assert not (tmp_path / "a.csv.prefilter.bak").exists()  # no backup written


def test_cli_dry_run_multi_file_leaves_files_unchanged(tmp_path):
    import pandas as pd
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    pd.DataFrame(_ROWS).to_csv(a, index=False)
    pd.DataFrame(_ROWS).to_csv(b, index=False)
    cli_main([str(a), str(b)])  # dry run
    assert len(pd.read_csv(a)) == 3 and len(pd.read_csv(b)) == 3
