"""Versioned tensor contract for the Phase 2 raw-XIC dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spectrum.labeling import IDEAL_FULL_LABEL_ISOTOPE_MODEL

SCHEMA_VERSION = "phase2_raw_xic_v3"
PRECURSOR_CHANNELS = ("light_m0", "heavy_m0", "heavy_m1", "heavy_m2")
PAIR_CHANNELS = ("light", "heavy")
ION_TYPE_TO_CODE = {"b": 0, "y": 1}
ION_CODE_TO_TYPE = {value: key for key, value in ION_TYPE_TO_CODE.items()}

FRAGMENT_STATUS_TO_CODE = {
    "valid": 0,
    "no_light_peak": 1,
    "no_heavy_peak": 2,
    "no_light_or_heavy_peak": 3,
    "coisolated_same_mz": 4,
    "heavy_out_of_range": 5,
}
FRAGMENT_CODE_TO_STATUS = {
    value: key for key, value in FRAGMENT_STATUS_TO_CODE.items()
}


@dataclass(frozen=True)
class ExtractionSettings:
    """Signal extraction settings that participate in dataset identity."""

    xic_cycle_window: int = 6
    mass_tol_ppm: float = 10.0
    fragment_charges: tuple[int, ...] = (1, 2)

    def __post_init__(self) -> None:
        if self.xic_cycle_window < 0:
            raise ValueError("xic_cycle_window must be non-negative")
        if not np.isfinite(self.mass_tol_ppm) or self.mass_tol_ppm <= 0:
            raise ValueError("mass_tol_ppm must be a positive finite value")
        charges = tuple(int(value) for value in self.fragment_charges)
        if not charges or any(value <= 0 for value in charges):
            raise ValueError("fragment_charges must be positive integers")
        if len(set(charges)) != len(charges):
            raise ValueError("fragment_charges must be unique")
        object.__setattr__(self, "fragment_charges", charges)

    @property
    def trace_length(self) -> int:
        return 2 * self.xic_cycle_window + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "xic_cycle_window": self.xic_cycle_window,
            "trace_length": self.trace_length,
            "mass_tol_ppm": self.mass_tol_ppm,
            "fragment_charges": list(self.fragment_charges),
        }


@dataclass
class SignalSample:
    """One PSM's lossless fixed-window precursor and ragged fragment traces."""

    metadata: dict[str, Any]
    precursor_intensity: np.ndarray
    precursor_ppm_error: np.ndarray
    precursor_rt_delta: np.ndarray
    precursor_scan_mask: np.ndarray
    precursor_peak_mask: np.ndarray
    fragment_intensity: np.ndarray
    fragment_ppm_error: np.ndarray
    fragment_rt_delta: np.ndarray
    fragment_scan_mask: np.ndarray
    fragment_peak_mask: np.ndarray
    fragment_ion_type: np.ndarray
    fragment_ordinal: np.ndarray
    fragment_charge: np.ndarray
    fragment_light_mz: np.ndarray
    fragment_heavy_mz: np.ndarray
    fragment_predicted_intensity: np.ndarray
    fragment_prediction_present: np.ndarray
    fragment_separable: np.ndarray
    fragment_attempted: np.ndarray
    fragment_status: np.ndarray

    def validate(self, settings: ExtractionSettings) -> None:
        """Reject malformed samples before any partial shard is published."""
        if not str(self.metadata.get("sample_id", "")):
            raise ValueError("sample metadata requires a non-empty sample_id")
        if self.metadata.get("label") not in (0, 1):
            raise ValueError("stored label must be 1=correct or 0=incorrect")
        for name in (
            "center_cycle", "fragment_light_center_cycle",
            "fragment_heavy_center_cycle",
        ):
            if name not in self.metadata:
                raise ValueError(f"sample metadata requires {name}")
            try:
                int(self.metadata[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sample metadata {name} must be an integer") from exc

        trace = settings.trace_length
        precursor_shape = (len(PRECURSOR_CHANNELS), trace)
        for name in (
            "precursor_intensity", "precursor_ppm_error",
            "precursor_rt_delta", "precursor_scan_mask",
            "precursor_peak_mask",
        ):
            if np.shape(getattr(self, name)) != precursor_shape:
                raise ValueError(
                    f"{name} has shape {np.shape(getattr(self, name))}; "
                    f"expected {precursor_shape}")

        n_fragments = int(len(self.fragment_ion_type))
        trace_shape = (n_fragments, len(PAIR_CHANNELS), trace)
        for name in (
            "fragment_intensity", "fragment_ppm_error",
            "fragment_rt_delta", "fragment_scan_mask", "fragment_peak_mask",
        ):
            if np.shape(getattr(self, name)) != trace_shape:
                raise ValueError(
                    f"{name} has shape {np.shape(getattr(self, name))}; "
                    f"expected {trace_shape}")
        for name in (
            "fragment_ordinal", "fragment_charge", "fragment_light_mz",
            "fragment_heavy_mz", "fragment_predicted_intensity",
            "fragment_prediction_present", "fragment_separable",
            "fragment_attempted", "fragment_status",
        ):
            if len(getattr(self, name)) != n_fragments:
                raise ValueError(f"{name} length differs from fragment count")

        for name in ("precursor_intensity", "fragment_intensity"):
            values = np.asarray(getattr(self, name))
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "precursor_ppm_error", "precursor_rt_delta",
            "fragment_ppm_error", "fragment_rt_delta",
        ):
            if not np.isfinite(np.asarray(getattr(self, name))).all():
                raise ValueError(f"{name} must use masks, not NaN/Inf")

        if not set(np.unique(self.fragment_ion_type)).issubset(
                ION_CODE_TO_TYPE):
            raise ValueError("fragment_ion_type contains an unknown code")
        if not set(np.unique(self.fragment_status)).issubset(
                FRAGMENT_CODE_TO_STATUS):
            raise ValueError("fragment_status contains an unknown code")
        present = np.asarray(self.fragment_prediction_present, dtype=bool)
        predicted = np.asarray(self.fragment_predicted_intensity)
        if present.any() and not np.isfinite(predicted[present]).all():
            raise ValueError("present spectral predictions must be finite")
        if (~present).any() and not np.isnan(predicted[~present]).all():
            raise ValueError("missing spectral predictions must use NaN")


def schema_document(settings: ExtractionSettings) -> dict[str, Any]:
    """Return the complete machine-readable storage and model-use contract."""
    return {
        "schema": SCHEMA_VERSION,
        "storage_label": "1=correct_identification,0=incorrect_identification",
        "model_score": "trust_score=P(correct_identification)",
        "metric_semantics": "error_identification_positive_v1",
        "positive_class": "incorrect_identification",
        "isotope_model": IDEAL_FULL_LABEL_ISOTOPE_MODEL,
        "isotope_channel_contract": (
            "nominal M0/M1/M2 channels union-match chemistry-aware exact-mass "
            "isotopologue targets without double-counting centroids"
        ),
        "precursor_channels": list(PRECURSOR_CHANNELS),
        "fragment_pair_channels": list(PAIR_CHANNELS),
        "ion_type_codes": ION_TYPE_TO_CODE,
        "fragment_status_codes": FRAGMENT_STATUS_TO_CODE,
        "extraction": settings.to_dict(),
        "mask_contract": {
            "scan_mask": "1=real acquired scan,0=padding or unavailable",
            "peak_mask": "1=matched peak with finite ppm error",
            "zero_intensity_with_scan_mask_1": "real scan with no matched peak",
        },
        "cycle_alignment_contract": {
            "precursor_center": "nearest_ms1_cycle_to_psm_rt",
            "fragment_light_center": (
                "selected_light_precursor_ms2_isolation_window_cycle"),
            "fragment_heavy_center": (
                "selected_heavy_precursor_ms2_isolation_window_cycle"),
            "out_of_range_scan_policy": "error_never_silently_truncate",
        },
        "model_input_policy": {
            "signal_derived_global_features": [
                "log1p_sample_max_intensity",
                "log1p_sample_abs_max_rt_delta",
            ],
            "allowed": [
                "intensity", "ppm_error", "rt_delta", "scan_mask",
                "peak_mask", "fragment ion_type/charge as theoretical-ion "
                "context", "attempted/separable only as a hard attention "
                "eligibility gate",
                "optional predicted_intensity only in declared prediction arm",
            ],
            "audit_only": [
                "sequence", "protein_names", "label_type", "negative_tier",
                "q_value", "raw_title", "dataset", "fixed_split",
                "outer_fold", "inner_valid_for_fold_*", "absolute_rt",
                "precursor_charge", "fragment_ordinal", "fragment_count",
            ],
        },
    }
