"""Recompute selected legacy features from stored Phase 2 tensors."""

from __future__ import annotations

import json

import numpy as np

from spectrum.dia_data import XIC_DTYPE
from spectrum.labeling import (
    COMPATIBLE_LEGACY_ISOTOPE_MODELS, IDEAL_FULL_LABEL_ISOTOPE_MODEL,
)
from spectrum.psm_info import get_theoretical_isotope_ratios
from workflows.single_work import (
    _is_empty_xic_pair, calc_xic_score, extract_ion_pearson_features,
)

from .schema import (
    FRAGMENT_STATUS_TO_CODE, ION_CODE_TO_TYPE, ExtractionSettings,
    SignalSample,
)


PARITY_FEATURES = (
    "precursor_pearson", "precursor_cosine", "precursor_apex_delta",
    "precursor_mz_avg_err", "isotope_correlation", "mass_shift_error",
    "b_count", "b_mean", "y_count", "y_mean",
    "all_count", "all_mean", "frag_corr_weighted",
    "valid_fragment_ions_num", "fragment_xic_empty_count",
    "fragment_heavy_absent_count", "fragment_same_mass_count",
)
_ISOTOPE_MODEL_DEPENDENT_FEATURES = frozenset({"isotope_correlation"})
_AUDIT_ONLY_LEGACY_ISOTOPE_MODELS = frozenset({
    "", *COMPATIBLE_LEGACY_ISOTOPE_MODELS,
})


def _decode_xic(intensity, ppm_error, rt_delta, scan_mask, peak_mask,
                *, center_rt: float, center_cycle: int,
                settings: ExtractionSettings) -> np.ndarray:
    rows = []
    for slot in np.flatnonzero(scan_mask):
        cycle_idx = center_cycle + int(slot) - settings.xic_cycle_window
        ppm = float(ppm_error[slot]) if peak_mask[slot] else float("nan")
        rows.append((
            center_rt + float(rt_delta[slot]), ppm,
            float(intensity[slot]), cycle_idx,
        ))
    return np.asarray(rows, dtype=XIC_DTYPE)


def _pool_fragment_group(sample: SignalSample, indices: list[int], side: int,
                         settings: ExtractionSettings) -> np.ndarray:
    first = indices[0]
    scan_mask = np.asarray(sample.fragment_scan_mask[first, side], bool)
    # Charge-resolved extraction uses one selected scan set.  A mismatch is
    # schema corruption, not something parity should silently reconcile.
    for index in indices[1:]:
        if not np.array_equal(
                scan_mask, sample.fragment_scan_mask[index, side]):
            raise ValueError("charge records have different scan masks")
    intensity = np.sum(
        sample.fragment_intensity[indices, side], axis=0, dtype="f8")
    ppm = np.zeros(settings.trace_length, dtype="f8")
    peak_mask = np.zeros(settings.trace_length, dtype=bool)
    for slot in np.flatnonzero(scan_mask):
        errors = sample.fragment_ppm_error[indices, side, slot]
        present = sample.fragment_peak_mask[indices, side, slot]
        if not np.any(present):
            continue
        weights = sample.fragment_intensity[indices, side, slot][present]
        ppm[slot] = (
            np.average(errors[present], weights=weights)
            if float(np.sum(weights)) > 0 else np.mean(errors[present])
        )
        peak_mask[slot] = True
    return _decode_xic(
        intensity, ppm, sample.fragment_rt_delta[first, side], scan_mask,
        peak_mask, center_rt=float(sample.metadata["rt"]),
        center_cycle=int(sample.metadata[
            "fragment_light_center_cycle" if side == 0
            else "fragment_heavy_center_cycle"
        ]), settings=settings)


def reconstruct_legacy_features(sample: SignalSample,
                                settings: ExtractionSettings) -> dict:
    """Reconstruct features that directly verify extraction equivalence."""
    center_rt = float(sample.metadata["rt"])
    center_cycle = int(sample.metadata["center_cycle"])
    precursor = [
        _decode_xic(
            sample.precursor_intensity[channel],
            sample.precursor_ppm_error[channel],
            sample.precursor_rt_delta[channel],
            sample.precursor_scan_mask[channel],
            sample.precursor_peak_mask[channel], center_rt=center_rt,
            center_cycle=center_cycle, settings=settings)
        for channel in (0, 1)
    ]
    if _is_empty_xic_pair(*precursor):
        result = {
            "precursor_pearson": 0.0, "precursor_cosine": 0.0,
            "precursor_apex_delta": 0.0, "precursor_mz_avg_err": 0.0,
        }
    else:
        score = calc_xic_score(*precursor, center_rt=center_rt)
        result = {
            "precursor_pearson": score["pearson"],
            "precursor_cosine": score["cosine"],
            "precursor_apex_delta": score["apex_delta"],
            "precursor_mz_avg_err": score["mz_avg_err"],
        }

    heavy_intensity = np.asarray(sample.precursor_intensity[1], dtype="f8")
    if not np.any(heavy_intensity > 0):
        result["isotope_correlation"] = float("nan")
        result["mass_shift_error"] = float("nan")
    else:
        apex_slot = int(np.argmax(heavy_intensity))
        observed = np.asarray([
            sample.precursor_intensity[channel, apex_slot]
            for channel in (1, 2, 3)
        ], dtype="f8")
        theoretical = np.asarray(get_theoretical_isotope_ratios(
            str(sample.metadata["sequence"]),
            json.loads(str(sample.metadata.get("modifications_json", "[]"))),
            str(sample.metadata["labeling"]),
        ), dtype="f8")
        denominator = np.linalg.norm(observed) * np.linalg.norm(theoretical)
        result["isotope_correlation"] = (
            float(np.dot(observed, theoretical) / denominator)
            if denominator > 0 else 0.0)
        result["mass_shift_error"] = (
            float(sample.precursor_ppm_error[1, apex_slot])
            if sample.precursor_peak_mask[1, apex_slot]
            else float("nan"))

    logical: dict[tuple[int, int], list[int]] = {}
    for index, (ion_type, ordinal) in enumerate(zip(
            sample.fragment_ion_type, sample.fragment_ordinal)):
        logical.setdefault((int(ion_type), int(ordinal)), []).append(index)
    pearsons = {"b": [], "y": [], "all": []}
    weights = []
    empty_count = same_mass_count = heavy_absent_count = attempted = 0
    for (ion_code, _ordinal), indices in logical.items():
        statuses = set(int(sample.fragment_status[index]) for index in indices)
        if statuses == {FRAGMENT_STATUS_TO_CODE["heavy_out_of_range"]}:
            heavy_absent_count += 1
            continue
        if statuses == {FRAGMENT_STATUS_TO_CODE["coisolated_same_mz"]}:
            same_mass_count += 1
            continue
        structural = {
            FRAGMENT_STATUS_TO_CODE["heavy_out_of_range"],
            FRAGMENT_STATUS_TO_CODE["coisolated_same_mz"],
        }
        if statuses & structural:
            raise ValueError(
                "fragment charges disagree on structural separability status")
        attempted += 1
        light = _pool_fragment_group(sample, indices, 0, settings)
        heavy = _pool_fragment_group(sample, indices, 1, settings)
        ion_type = ION_CODE_TO_TYPE[ion_code]
        if _is_empty_xic_pair(light, heavy):
            pearson = float("nan")
            weight = float("nan")
            empty_count += 1
        else:
            score = calc_xic_score(light, heavy, center_rt=center_rt)
            pearson = score["pearson"]
            weight = max(score["light_max_int"], score["heavy_max_int"])
        pearsons[ion_type].append(pearson)
        pearsons["all"].append(pearson)
        weights.append(weight)

    for ion_type in ("b", "y", "all"):
        stats = extract_ion_pearson_features(pearsons[ion_type])
        result[f"{ion_type}_count"] = stats["count"]
        result[f"{ion_type}_mean"] = stats["mean"]
    finite = [
        (pearson, weight) for pearson, weight in zip(pearsons["all"], weights)
        if np.isfinite(pearson) and np.isfinite(weight) and weight > 0
    ]
    total_weight = sum(weight for _, weight in finite)
    result["frag_corr_weighted"] = (
        sum(pearson * weight for pearson, weight in finite) / total_weight
        if total_weight > 0 else float("nan"))
    result.update({
        "valid_fragment_ions_num": attempted,
        "fragment_xic_empty_count": empty_count,
        "fragment_heavy_absent_count": heavy_absent_count,
        "fragment_same_mass_count": same_mass_count,
    })
    return result


def compare_to_feature_row(sample: SignalSample, feature_row,
                           settings: ExtractionSettings,
                           *, atol: float = 1e-4,
                           rtol: float = 1e-5) -> list[dict]:
    """Return value-level parity rows for all available pinned features."""
    reconstructed = reconstruct_legacy_features(sample, settings)
    rows = []
    snapshot_isotope_model = str(
        feature_row.get("isotope_model", "")).strip()
    if snapshot_isotope_model.lower() in {"", "nan", "none", "<na>"}:
        snapshot_isotope_model = ""
    for feature in PARITY_FEATURES:
        if feature not in feature_row:
            continue
        expected = float(feature_row[feature])
        observed = float(reconstructed[feature])
        both_nan = np.isnan(expected) and np.isnan(observed)
        passed = bool(both_nan or np.isclose(
            expected, observed, atol=atol, rtol=rtol, equal_nan=True))
        model_migration = (
            feature in _ISOTOPE_MODEL_DEPENDENT_FEATURES
            and snapshot_isotope_model in _AUDIT_ONLY_LEGACY_ISOTOPE_MODELS
        )
        rows.append({
            "sample_id": str(sample.metadata["sample_id"]),
            "feature": feature,
            "expected": expected,
            "reconstructed": observed,
            "absolute_difference": (
                abs(expected - observed)
                if np.isfinite(expected) and np.isfinite(observed)
                else float("nan")),
            "passed": passed,
            "required_for_publish": not model_migration,
            "parity_policy": (
                "legacy_isotope_model_audit_only" if model_migration
                else "required"
            ),
            "snapshot_isotope_model": (
                snapshot_isotope_model or "undeclared_legacy"
            ),
            "reconstructed_isotope_model": IDEAL_FULL_LABEL_ISOTOPE_MODEL,
        })
    return rows
