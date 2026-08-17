"""Extract one PSM into the versioned Phase 2 signal contract."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from spectrum.dia_data import DIAData, XIC_DTYPE
from spectrum.labeling import (
    HeavyType, canonical_labeling_name, parse_heavy_type,
)
from spectrum.psm_info import (
    PROTON_MASS, PSMInfo, get_isotopologue_mz_targets,
)
from workflows.pred_store import frag_key, frag_pos_for_ion
from workflows.q1a_helpers import SHIFT_EPSILON

from .schema import (
    FRAGMENT_STATUS_TO_CODE, ION_TYPE_TO_CODE, PRECURSOR_CHANNELS,
    ExtractionSettings, SignalSample,
)


def _encode_xic(
    xic: np.ndarray,
    *,
    center_cycle: int,
    center_rt: float,
    settings: ExtractionSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align a sparse/edge XIC onto global cycle offsets with two masks."""
    trace_length = settings.trace_length
    intensity = np.zeros(trace_length, dtype="f4")
    ppm_error = np.zeros(trace_length, dtype="f4")
    rt_delta = np.zeros(trace_length, dtype="f4")
    scan_mask = np.zeros(trace_length, dtype=bool)
    peak_mask = np.zeros(trace_length, dtype=bool)
    if xic is None:
        return intensity, ppm_error, rt_delta, scan_mask, peak_mask

    for row in xic:
        cycle_idx = int(row["cycle_idx"])
        slot = cycle_idx - center_cycle + settings.xic_cycle_window
        if slot < 0 or slot >= trace_length:
            raise ValueError(
                "XIC scan falls outside its declared center cycle: "
                f"cycle_idx={cycle_idx}, center_cycle={center_cycle}, "
                f"window={settings.xic_cycle_window}")
        if scan_mask[slot]:
            raise ValueError(
                f"XIC contains duplicate global cycle_idx={cycle_idx}")
        observed_intensity = float(row["intensity"])
        if not np.isfinite(observed_intensity) or observed_intensity < 0:
            raise ValueError("XIC intensity must be finite and non-negative")
        observed_ppm = float(row["ppm_error"])
        intensity[slot] = observed_intensity
        rt_delta[slot] = float(row["rt"]) - center_rt
        scan_mask[slot] = True
        if np.isfinite(observed_ppm):
            ppm_error[slot] = observed_ppm
            peak_mask[slot] = True
    return intensity, ppm_error, rt_delta, scan_mask, peak_mask


def _empty_xic() -> np.ndarray:
    return np.empty(0, dtype=XIC_DTYPE)


def _peak_status(light_xic: np.ndarray, heavy_xic: np.ndarray) -> str:
    """Describe peak presence for one charge-resolved fragment pair."""
    light_present = bool(
        len(light_xic) and np.any(light_xic["intensity"] > 0))
    heavy_present = bool(
        len(heavy_xic) and np.any(heavy_xic["intensity"] > 0))
    if light_present and heavy_present:
        return "valid"
    if not light_present and not heavy_present:
        return "no_light_or_heavy_peak"
    return "no_light_peak" if not light_present else "no_heavy_peak"


def _extract_ms1_panels(dia_data, rt, settings, panels):
    method = getattr(dia_data, "xic_peaks_panels_extract", None)
    if method is not None:
        return method(
            rt, settings.xic_cycle_window, panels, settings.mass_tol_ppm)
    return [
        dia_data.xic_peaks_panel_extract(
            rt, settings.xic_cycle_window, panel, settings.mass_tol_ppm)
        for panel in panels
    ]


def _extract_ms2_panel(dia_data, psm, settings, precursor_mz, masses):
    method = getattr(dia_data, "xic_ms2_fragment_panel_extract", None)
    if method is not None:
        values, _ = method(
            psm._rt, settings.xic_cycle_window, precursor_mz, masses,
            settings.mass_tol_ppm, settings.fragment_charges)
        return values
    return [
        dia_data.xic_ms2_charge_resolved_extract(
            psm._rt, settings.xic_cycle_window, precursor_mz, mass_value,
            settings.mass_tol_ppm, settings.fragment_charges)[0]
        for mass_value in masses
    ]


def _resolve_ms2_center_cycle(
    dia_data, *, rt: float, precursor_mz: float, fallback: int,
) -> int:
    """Resolve the actual center used by the MS2 isolation-window selector."""
    method = getattr(dia_data, "resolve_ms2_xic_center_cycle", None)
    if method is None:
        # Lightweight third-party/test DIA stand-ins predate the explicit
        # center API and historically align their traces to the MS1 center.
        return int(fallback)
    center = method(rt, precursor_mz)
    return int(fallback if center is None else center)


def extract_signal_sample(
    psm: PSMInfo,
    dia_data: DIAData,
    settings: ExtractionSettings,
    metadata: dict[str, Any],
    *,
    labeling: HeavyType | str = HeavyType.SILAC,
    pred_frags: dict | None = None,
) -> SignalSample:
    """Extract precursor/isotope and charge-resolved fragment XICs.

    Absolute RT, sequence, labels and split information stay in ``metadata``;
    only relative time traces are placed in model-facing arrays.
    """
    selected_labeling = parse_heavy_type(labeling)
    center_rt = float(psm._rt)
    if dia_data.ms1_indexs is None or len(dia_data.ms1_indexs) == 0:
        raise ValueError("raw acquisition has no MS1 scans")
    center_cycle = int(dia_data.find_near_ms1_idx(psm._rt))
    heavy_precursor_mz, fragment_ions = psm.get_heavy_info(
        selected_labeling)

    isotope_targets = get_isotopologue_mz_targets(
        heavy_precursor_mz, psm._charge, psm._sequence, psm._modify,
        selected_labeling)
    precursor_mz = (
        [float(psm._precursor_mz)], isotope_targets[0],
        isotope_targets[1], isotope_targets[2],
    )
    precursor_arrays = [[], [], [], [], []]
    for xic in _extract_ms1_panels(
            dia_data, psm._rt, settings, precursor_mz):
        encoded = _encode_xic(
            xic, center_cycle=center_cycle, center_rt=center_rt,
            settings=settings)
        for values, value in zip(precursor_arrays, encoded):
            values.append(value)

    fragment_values: dict[str, list] = {
        "intensity": [], "ppm_error": [], "rt_delta": [],
        "scan_mask": [], "peak_mask": [], "ion_type": [],
        "ordinal": [], "charge": [], "light_mz": [], "heavy_mz": [],
        "predicted_intensity": [], "prediction_present": [],
        "separable": [], "attempted": [], "status": [],
    }
    heavy_in_raw = bool(dia_data.check_in_raw(heavy_precursor_mz))
    same_window = bool(dia_data.check_in_same_ms2(
        psm._precursor_mz, heavy_precursor_mz, rt=psm._rt))

    light_panels = _extract_ms2_panel(
        dia_data, psm, settings, psm._precursor_mz,
        [ion[2] for ion in fragment_ions])
    light_fragment_center_cycle = _resolve_ms2_center_cycle(
        dia_data, rt=psm._rt, precursor_mz=psm._precursor_mz,
        fallback=center_cycle)
    heavy_panels = (
        _extract_ms2_panel(
            dia_data, psm, settings, heavy_precursor_mz,
            [ion[3] for ion in fragment_ions])
        if heavy_in_raw else None
    )
    heavy_fragment_center_cycle = _resolve_ms2_center_cycle(
        dia_data, rt=psm._rt, precursor_mz=heavy_precursor_mz,
        fallback=center_cycle)

    for ion_index, (
            ion_type, ordinal, light_mass, heavy_mass) in enumerate(
                fragment_ions):
        coisolated = (
            abs(float(heavy_mass) - float(light_mass)) < SHIFT_EPSILON
            and same_window
        )
        light_by_charge = light_panels[ion_index]
        if not heavy_in_raw:
            heavy_by_charge = {
                charge: _empty_xic() for charge in settings.fragment_charges
            }
            status = "heavy_out_of_range"
            separable = False
            attempted = False
        elif coisolated:
            # It is one physical observation, retained for audit but masked
            # from paired-evidence models.
            heavy_by_charge = light_by_charge
            status = "coisolated_same_mz"
            separable = False
            attempted = False
        else:
            heavy_by_charge = heavy_panels[ion_index]
            separable = True
            attempted = True

        fragment_position = frag_pos_for_ion(
            ion_type, ordinal, len(psm._sequence))
        for charge in settings.fragment_charges:
            charge_status = (
                status if not attempted else _peak_status(
                    light_by_charge.get(charge, _empty_xic()),
                    heavy_by_charge.get(charge, _empty_xic()))
            )
            encoded_pair = [
                _encode_xic(
                    by_charge.get(charge, _empty_xic()),
                    center_cycle=fragment_center_cycle,
                    center_rt=center_rt,
                    settings=settings)
                for by_charge, fragment_center_cycle in zip(
                    (light_by_charge, heavy_by_charge),
                    (light_fragment_center_cycle,
                     heavy_fragment_center_cycle),
                )
            ]
            fragment_values["intensity"].append(np.stack(
                [encoded[0] for encoded in encoded_pair]))
            fragment_values["ppm_error"].append(np.stack(
                [encoded[1] for encoded in encoded_pair]))
            fragment_values["rt_delta"].append(np.stack(
                [encoded[2] for encoded in encoded_pair]))
            fragment_values["scan_mask"].append(np.stack(
                [encoded[3] for encoded in encoded_pair]))
            fragment_values["peak_mask"].append(np.stack(
                [encoded[4] for encoded in encoded_pair]))
            fragment_values["ion_type"].append(ION_TYPE_TO_CODE[ion_type])
            fragment_values["ordinal"].append(int(ordinal))
            fragment_values["charge"].append(int(charge))
            fragment_values["light_mz"].append(
                (float(light_mass) + charge * PROTON_MASS) / charge)
            fragment_values["heavy_mz"].append(
                (float(heavy_mass) + charge * PROTON_MASS) / charge)
            predicted = None if pred_frags is None else pred_frags.get(
                frag_key(ion_type, fragment_position, charge))
            prediction_present = (
                predicted is not None and np.isfinite(float(predicted)))
            fragment_values["predicted_intensity"].append(
                float(predicted) if prediction_present else float("nan"))
            fragment_values["prediction_present"].append(prediction_present)
            fragment_values["separable"].append(separable)
            fragment_values["attempted"].append(attempted)
            fragment_values["status"].append(
                FRAGMENT_STATUS_TO_CODE[charge_status])

    n_fragments = len(fragment_values["ion_type"])
    trace_shape = (0, 2, settings.trace_length)

    def traces(name: str, dtype) -> np.ndarray:
        values = fragment_values[name]
        return (np.asarray(values, dtype=dtype) if values
                else np.empty(trace_shape, dtype=dtype))

    sample_metadata = dict(metadata)
    sample_metadata.update({
        "labeling": canonical_labeling_name(selected_labeling),
        "center_cycle": center_cycle,
        "fragment_light_center_cycle": light_fragment_center_cycle,
        "fragment_heavy_center_cycle": heavy_fragment_center_cycle,
        "heavy_precursor_mz": float(heavy_precursor_mz),
        "heavy_in_raw": int(heavy_in_raw),
        "precursor_channels": json.dumps(PRECURSOR_CHANNELS),
        "isotope_m1_target_mz_json": json.dumps(
            isotope_targets[1], separators=(",", ":")),
        "isotope_m2_target_mz_json": json.dumps(
            isotope_targets[2], separators=(",", ":")),
    })
    sample = SignalSample(
        metadata=sample_metadata,
        precursor_intensity=np.asarray(precursor_arrays[0], dtype="f4"),
        precursor_ppm_error=np.asarray(precursor_arrays[1], dtype="f4"),
        precursor_rt_delta=np.asarray(precursor_arrays[2], dtype="f4"),
        precursor_scan_mask=np.asarray(precursor_arrays[3], dtype=bool),
        precursor_peak_mask=np.asarray(precursor_arrays[4], dtype=bool),
        fragment_intensity=traces("intensity", "f4"),
        fragment_ppm_error=traces("ppm_error", "f4"),
        fragment_rt_delta=traces("rt_delta", "f4"),
        fragment_scan_mask=traces("scan_mask", bool),
        fragment_peak_mask=traces("peak_mask", bool),
        fragment_ion_type=np.asarray(
            fragment_values["ion_type"], dtype="i1"),
        fragment_ordinal=np.asarray(
            fragment_values["ordinal"], dtype="i2"),
        fragment_charge=np.asarray(
            fragment_values["charge"], dtype="i1"),
        fragment_light_mz=np.asarray(
            fragment_values["light_mz"], dtype="f4"),
        fragment_heavy_mz=np.asarray(
            fragment_values["heavy_mz"], dtype="f4"),
        fragment_predicted_intensity=np.asarray(
            fragment_values["predicted_intensity"], dtype="f4"),
        fragment_prediction_present=np.asarray(
            fragment_values["prediction_present"], dtype=bool),
        fragment_separable=np.asarray(
            fragment_values["separable"], dtype=bool),
        fragment_attempted=np.asarray(
            fragment_values["attempted"], dtype=bool),
        fragment_status=np.asarray(
            fragment_values["status"], dtype="u1"),
    )
    if len(sample.fragment_ion_type) != n_fragments:
        raise AssertionError("fragment extraction lost metadata rows")
    sample.validate(settings)
    return sample
