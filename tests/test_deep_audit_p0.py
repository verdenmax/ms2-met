"""Phase 0 (Critical) behavioral tests for deep audit fixes.

See docs/specs/2026-06-03-deep-audit-fixes-design.md for design rationale.
"""
import configparser
import numpy as np
import pytest


def _empty_xic():
    """Return an empty XIC structured array matching dia_data dtype."""
    dtype = [("rt", "f8"), ("ppm_error", "f8"),
             ("intensity", "f8"), ("cycle_idx", "i4")]
    return np.array([], dtype=dtype)


def _real_xic(rts, intensities, cycle_idxs=None):
    """Build a non-empty XIC for tests."""
    n = len(rts)
    if cycle_idxs is None:
        cycle_idxs = list(range(n))
    dtype = [("rt", "f8"), ("ppm_error", "f8"),
             ("intensity", "f8"), ("cycle_idx", "i4")]
    arr = np.zeros(n, dtype=dtype)
    arr["rt"] = rts
    arr["ppm_error"] = 0.0
    arr["intensity"] = intensities
    arr["cycle_idx"] = cycle_idxs
    return arr


class _FakePSM:
    """Minimal PSM stub for triggering single_pair_work / multi_batch_work."""
    def __init__(self, mz=500.0, rt=10.0, seq="AAAAK", charge=2):
        self._precursor_mz = mz
        self._rt = rt
        self._sequence = seq
        self._charge = charge
        self._raw_title = "fake.mzML"
        self._protein_names = "HUMAN"
        self._label_type = "positive"
        self._modify = []

    def get_heavy_info(self, heavy_type):
        # Synthetic fragment_ions: 2 b-ions + 2 y-ions with non-zero SILAC shift
        # so they are NOT skipped by the same-mass guard in single_pair_work.
        heavy_mz = self._precursor_mz + 4.0
        fragment_ions = [
            ("b", 1, 100.0, 108.014),
            ("b", 2, 200.0, 208.014),
            ("y", 1, 150.0, 158.014),
            ("y", 2, 250.0, 258.014),
        ]
        return heavy_mz, fragment_ions


class _FakeDIA:
    """Minimal DIA stub. Returns empty XIC for any precursor_mz query
    when force_empty=True; otherwise returns a small synthetic XIC."""
    def __init__(self, force_empty=False, xic_intensity=None):
        self._force_empty = force_empty
        self._xic_intensity = xic_intensity
        self._min_mz_value = 0.0
        self._max_mz_value = 10000.0

    def xic_peaks_extreact(self, rt, window, mz, mass_tol_ppm):
        if self._force_empty:
            return _empty_xic()
        intensity = self._xic_intensity if self._xic_intensity is not None \
            else [100.0, 200.0, 500.0, 300.0, 150.0]
        return _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], intensity)

    def xic_ms2_peaks_extract(self, rt, window, precursor_mz, ions_mass,
                              mass_tol_ppm):
        if self._force_empty:
            return _empty_xic(), 0.0
        intensity = self._xic_intensity if self._xic_intensity is not None \
            else [10.0, 20.0, 50.0, 30.0, 15.0]
        return _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], intensity), 100.0

    def check_in_raw(self, mz):
        return True

    def check_in_same_ms2(self, p1, p2):
        return False

    def get_window_info(self, mz):
        return {"width": 2.0, "centering": 0.5,
                "lower": mz - 1.0, "upper": mz + 1.0,
                "split_window": False}


def _minimal_config():
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "general": {
            "mass_tol_ppm": "20",
            "xic_cycle_window": "5",
        },
    })
    return cfg


def test_single_pair_work_marks_precursor_xic_empty_when_empty():
    """single_pair_work empty-XIC branch must set precursor_xic_empty=1."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=True)
    features = single_pair_work(psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 1, (
        "P0-1: empty-XIC must set marker=1 in single_pair_work")


def test_multi_batch_work_marks_precursor_xic_empty_when_empty():
    """multi_batch_work empty-XIC branch must set precursor_xic_empty=1."""
    from workflows.single_work import multi_batch_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=True)
    features = multi_batch_work(psm, dia, psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 1, (
        "P0-1: empty-XIC must set marker=1 in multi_batch_work")


def test_single_pair_work_marks_precursor_xic_empty_zero_when_valid():
    """Non-empty valid XIC must set precursor_xic_empty=0 (single_pair)."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    features = single_pair_work(psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 0, (
        "P0-1: valid XIC must set marker=0 in single_pair_work")


def test_multi_batch_work_marks_precursor_xic_empty_zero_when_valid():
    """Non-empty valid XIC must set precursor_xic_empty=0 (multi_batch)."""
    from workflows.single_work import multi_batch_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    features = multi_batch_work(psm, dia, psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 0, (
        "P0-1: valid XIC must set marker=0 in multi_batch_work")


def test_fragment_xic_empty_count_present_in_both_paths():
    """Both code paths must emit all 3 fragment-skip-cause columns."""
    from workflows.single_work import single_pair_work, multi_batch_work
    psm = _FakePSM()
    dia_empty = _FakeDIA(force_empty=True)
    f1 = single_pair_work(psm, dia_empty, _minimal_config())
    f2 = multi_batch_work(psm, dia_empty, psm, dia_empty, _minimal_config())
    for key in ("fragment_xic_empty_count",
                "fragment_heavy_absent_count",
                "fragment_same_mass_count"):
        assert key in f1, (
            f"P0-1: single_pair_work missing {key}")
        assert key in f2, (
            f"P0-1: multi_batch_work missing {key}")


class _FakePSM_4Frags:
    """PSM stub returning 4 fragments with different mass shifts so we can
    count skip behavior precisely. Override _FakePSM.get_heavy_info."""
    def __init__(self, mz=500.0, rt=10.0):
        self._precursor_mz = mz
        self._rt = rt
        self._sequence = "AAAAK"
        self._charge = 2
        self._raw_title = "fake.mzML"
        self._protein_names = "HUMAN"
        self._label_type = "positive"
        self._modify = []

    def get_heavy_info(self, heavy_type):
        # 4 SILAC y-ions, each with non-zero mass shift between light/heavy
        return self._precursor_mz + 4.0, [
            ("y", 1, 100.0, 108.0),
            ("y", 2, 200.0, 208.0),
            ("y", 3, 300.0, 308.0),
            ("y", 4, 400.0, 408.0),
        ]


class _FakeDIA_NoHeavy(_FakeDIA):
    """DIA where heavy precursor is outside raw range — triggers
    heavy_in_raw=False fragment-skip path in single_pair_work."""
    def check_in_raw(self, mz):
        return False


def test_single_pair_work_counts_empty_xic_fragments():
    """fragment_xic_empty_count == number of fragments with empty XIC."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM_4Frags()
    # heavy_in_raw=True so we get past that guard; force_empty so XIC
    # extraction returns empty for all 4 fragments.
    dia = _FakeDIA(force_empty=True)
    features = single_pair_work(psm, dia, _minimal_config())
    assert features["fragment_xic_empty_count"] == 4, (
        f"Expected 4 empty-XIC fragments, got {features['fragment_xic_empty_count']}")
    assert features["fragment_heavy_absent_count"] == 0
    assert features["fragment_same_mass_count"] == 0


def test_single_pair_work_counts_heavy_absent_fragments():
    """fragment_heavy_absent_count == number of fragments skipped via
    heavy_in_raw=False."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM_4Frags()
    dia = _FakeDIA_NoHeavy(force_empty=False)
    features = single_pair_work(psm, dia, _minimal_config())
    assert features["fragment_heavy_absent_count"] == 4, (
        f"Expected 4 heavy_absent fragments, got "
        f"{features['fragment_heavy_absent_count']}")
    assert features["fragment_xic_empty_count"] == 0
    assert features["fragment_same_mass_count"] == 0


def test_multi_batch_work_heavy_absent_count_always_zero():
    """multi_batch_work has no heavy_in_raw guard so heavy_absent_count
    is always 0 (schema parity column)."""
    from workflows.single_work import multi_batch_work
    psm = _FakePSM_4Frags()
    dia = _FakeDIA(force_empty=False)
    features = multi_batch_work(psm, dia, psm, dia, _minimal_config())
    assert features["fragment_heavy_absent_count"] == 0
    assert features["fragment_same_mass_count"] == 0


def test_calc_xic_score_short_circuits_on_all_zero_intensity():
    """All-zero non-empty XIC must return _default_xic_score (P0-2, Silent-C2)."""
    from workflows.single_work import calc_xic_score, _default_xic_score
    light = _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], [0, 0, 0, 0, 0])
    heavy = _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], [0, 0, 0, 0, 0])
    result = calc_xic_score(light, heavy)
    default = _default_xic_score()
    # All keys in default should be present in result with the default value.
    for k, v in default.items():
        assert result[k] == v, (
            f"P0-2: all-zero XIC must produce default for {k}: "
            f"got {result[k]}, expected {v}")


def test_calc_xic_score_unchanged_on_valid_input():
    """Valid non-empty XIC should still produce computed (non-default) features."""
    from workflows.single_work import calc_xic_score
    light = _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], [10, 20, 100, 30, 15])
    heavy = _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], [5, 10, 50, 15, 8])
    result = calc_xic_score(light, heavy)
    # Both peaks at rt=10.0 -> apex_delta should be exactly 0 (real value).
    # Sanity-check pearson is high since shape is identical.
    assert result["pearson"] > 0.9, (
        f"P0-2: valid XIC should compute real pearson, got {result['pearson']}")
    assert result["light_max_int"] == 100.0
    assert result["heavy_max_int"] == 50.0


def test_single_pair_work_marks_precursor_xic_empty_on_all_zero_xic():
    """All-zero non-empty precursor XIC routes to empty branch via
    _is_empty_xic_pair → marker=1 (P0-1+P0-2 interaction)."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False, xic_intensity=[0, 0, 0, 0, 0])
    features = single_pair_work(psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 1, (
        "P0-1+P0-2: all-zero XIC must trigger marker=1")


def _populate_minimal_dia(dia):
    """Populate a DIAData instance with the minimal attrs save_to_file needs."""
    dia.has_mobility = False
    dia.has_ms1 = True
    dia._max_mz_value = 1000.0
    dia._min_mz_value = 100.0
    dia.ms1_indexs = np.array([0], dtype=np.int64)
    dia.ms1_indexs_rt = np.array([0.0])
    dia.ms2_indexs = np.array([1], dtype=np.int64)
    dia.ms2_indexs_rt = np.array([0.1])
    dia.precursor_scan_ids = np.array([0], dtype=np.int64)
    dia._mz_values = np.array([500.0])
    dia.rt_values = np.array([0.0, 0.1])
    dia._intensity_values = np.array([100.0])
    dia.mobility_values = np.array([])
    dia._cycle_left_precursor = np.array([400.0])
    dia._quad_max_mz_value = np.array([600.0])
    dia._quad_min_mz_value = np.array([400.0])
    dia._scan_id_to_index = np.array([0, 1], dtype=np.int64)
    dia._peak_start_idx_list = np.array([0], dtype=np.int64)
    dia._peak_stop_idx_list = np.array([1], dtype=np.int64)
    dia._precursor_lower_mz = np.array([400.0])
    dia._precursor_upper_mz = np.array([600.0])
    return dia


def test_cache_load_rejects_mismatched_centroid_threshold(tmp_path):
    """Cache saved with one rel_threshold must be rejected when loaded with
    a different expected_centroid_rel_threshold (P0-3, Silent-C3)."""
    from spectrum.dia_data import DIAData
    src = DIAData()
    src._centroid_enabled = True
    src._centroid_rel_threshold = 1e-3
    _populate_minimal_dia(src)
    cache_path = str(tmp_path / "cache_thresh.npz")
    src.save_to_file(cache_path)

    with pytest.raises(ValueError, match="centroid"):
        DIAData.load_from_file(cache_path, use_mmap=False,
                                expected_centroid_enabled=True,
                                expected_centroid_rel_threshold=1e-2)


def test_cache_load_rejects_mismatched_centroid_enabled(tmp_path):
    """Cache saved with centroid_enabled=True must be rejected if expected=False."""
    from spectrum.dia_data import DIAData
    src = DIAData()
    src._centroid_enabled = True
    src._centroid_rel_threshold = 1e-3
    _populate_minimal_dia(src)
    cache_path = str(tmp_path / "cache_enabled.npz")
    src.save_to_file(cache_path)

    with pytest.raises(ValueError, match="centroid"):
        DIAData.load_from_file(cache_path, use_mmap=False,
                                expected_centroid_enabled=False,
                                expected_centroid_rel_threshold=1e-3)


def test_cache_load_accepts_matching_centroid_params(tmp_path):
    """Cache with matching params loads successfully and restores params."""
    from spectrum.dia_data import DIAData
    src = DIAData()
    src._centroid_enabled = True
    src._centroid_rel_threshold = 1e-3
    _populate_minimal_dia(src)
    cache_path = str(tmp_path / "cache_match.npz")
    src.save_to_file(cache_path)

    loaded = DIAData.load_from_file(cache_path, use_mmap=False,
                                     expected_centroid_enabled=True,
                                     expected_centroid_rel_threshold=1e-3)
    assert loaded._centroid_enabled is True
    assert loaded._centroid_rel_threshold == 1e-3


def test_cache_load_no_expectations_accepts_any_params(tmp_path):
    """When neither expected_* arg is given, accept cache (back-compat for
    workers that trust the cache rather than re-validating)."""
    from spectrum.dia_data import DIAData
    src = DIAData()
    src._centroid_enabled = False  # arbitrary
    src._centroid_rel_threshold = 5e-4
    _populate_minimal_dia(src)
    cache_path = str(tmp_path / "cache_noexp.npz")
    src.save_to_file(cache_path)

    loaded = DIAData.load_from_file(cache_path, use_mmap=False)
    assert loaded._centroid_enabled is False
    assert loaded._centroid_rel_threshold == 5e-4


def test_cache_load_rejects_old_v2_format(tmp_path):
    """Old _format_version=2 caches (no centroid params) are rejected."""
    from spectrum.dia_data import DIAData
    cache_path = str(tmp_path / "v2_cache.npz")
    np.savez(cache_path, _format_version=np.int32(2))

    with pytest.raises(ValueError, match="_format_version"):
        DIAData.load_from_file(cache_path, use_mmap=False)
