"""Fold-local numeric preprocessing for neural models."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class FoldPreprocessor:
    """Median-impute and standardize using one training fold only.

    Missingness indicators are emitted for every source feature so all folds
    retain exactly the same model interface.  The fitted object is serialized
    with each checkpoint and must be reused for external predictions.
    """

    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    add_missing_indicators: bool = True

    @classmethod
    def fit(
        cls,
        values,
        *,
        add_missing_indicators: bool = True,
        scale_floor: float = 1e-6,
    ) -> "FoldPreprocessor":
        array = _as_2d_float(values)
        if np.isinf(array).any():
            raise ValueError("tabular features contain infinite values")
        if scale_floor <= 0:
            raise ValueError("scale_floor must be positive")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered",
                category=RuntimeWarning)
            medians = np.nanmedian(array, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = np.where(np.isnan(array), medians, array)
        means = imputed.mean(axis=0)
        scales = imputed.std(axis=0)
        scales = np.where(
            np.isfinite(scales) & (scales >= scale_floor), scales, 1.0)
        return cls(
            medians=medians.astype("f8"),
            means=means.astype("f8"),
            scales=scales.astype("f8"),
            add_missing_indicators=bool(add_missing_indicators),
        )

    @property
    def n_source_features(self) -> int:
        return int(len(self.medians))

    @property
    def n_output_features(self) -> int:
        multiplier = 2 if self.add_missing_indicators else 1
        return self.n_source_features * multiplier

    def transform(self, values) -> np.ndarray:
        array = _as_2d_float(values)
        if array.shape[1] != self.n_source_features:
            raise ValueError(
                "feature count does not match fitted preprocessor: "
                f"{array.shape[1]} vs {self.n_source_features}")
        if np.isinf(array).any():
            raise ValueError("tabular features contain infinite values")
        missing = np.isnan(array)
        imputed = np.where(missing, self.medians, array)
        standardized = (imputed - self.means) / self.scales
        if self.add_missing_indicators:
            standardized = np.concatenate(
                [standardized, missing.astype("f8")], axis=1)
        result = standardized.astype("f4", copy=False)
        if not np.isfinite(result).all():
            raise ValueError("preprocessed features are not finite")
        return result

    def to_state(self) -> dict:
        return {
            "schema": "fold_preprocessor_v1",
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
            "add_missing_indicators": self.add_missing_indicators,
        }

    @classmethod
    def from_state(cls, state: dict) -> "FoldPreprocessor":
        if state.get("schema") != "fold_preprocessor_v1":
            raise ValueError(
                f"unsupported preprocessor schema: {state.get('schema')!r}")
        return cls(
            medians=np.asarray(state["medians"], dtype="f8"),
            means=np.asarray(state["means"], dtype="f8"),
            scales=np.asarray(state["scales"], dtype="f8"),
            add_missing_indicators=bool(
                state.get("add_missing_indicators", True)),
        )


def _as_2d_float(values) -> np.ndarray:
    array = np.asarray(values, dtype="f8")
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(
            f"expected a nonempty 2-D feature matrix, got {array.shape}")
    return array
