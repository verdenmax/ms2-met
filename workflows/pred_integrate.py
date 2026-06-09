"""Per-PSM speclib predicted-intensity features — Phase 2a (I1 only).

Pure function over (already-separable) per-fragment records + the PSM's
predicted fragments. The pipeline (single_pair_work) does the split-aware
separability filtering and centroiding upstream (spec v1.2 4.1.5); this
module only does the math. See spec 4.3 (I1) and 7 (per-ion-type metric).
"""
import logging

import numpy as np

from workflows.pred_features import spectral_angle, _weighted_pearson
from workflows.pred_store import frag_key, frag_pos_for_ion

logger = logging.getLogger(__name__)

# Fixed output schema: every PSM emits identical columns (LightGBM-safe).
I1_KEYS = (
    "spec_pattern_SA_b",
    "spec_pattern_SA_y",
    "spec_pattern_SA",
    "spec_pattern_LH_consistency",
    "n_fragments_in_F",
)


def _nan_i1() -> dict:
    d = {k: float("nan") for k in I1_KEYS}
    d["n_fragments_in_F"] = 0
    return d


def compute_speclib_i1(frag_records, pred_frags, top_k, seq_len) -> dict:
    """I1 intensity-pattern features for one PSM (spec 4.3, per ion-type 7).

    frag_records: list of dicts (already-separable fragments) with keys
        ion_type ('b'/'y'), ion_num (1-based), light_apex, heavy_apex.
    pred_frags:   {frag_key: intensity} for this PSM, or None (no coverage).
    Returns a fixed-key dict (I1_KEYS); NaN where undefined.
    """
    if not pred_frags or not frag_records:
        logger.debug("compute_speclib_i1: no pred_frags or no records -> NaN")
        return _nan_i1()

    cands = []
    for r in frag_records:
        fp = frag_pos_for_ion(r["ion_type"], r["ion_num"], seq_len)
        pi = pred_frags.get(frag_key(r["ion_type"], fp, 1))
        if pi is None or not np.isfinite(pi):
            continue
        cands.append({**r, "pred": float(pi)})
    if not cands:
        logger.debug("compute_speclib_i1: no fragment matched a prediction -> NaN")
        return _nan_i1()

    cands.sort(key=lambda r: r["pred"], reverse=True)
    F = cands[:top_k]

    def sa_for(ion_type):
        sub = [r for r in F if r["ion_type"] == ion_type]
        if len(sub) < 2:
            return float("nan")
        return spectral_angle([r["pred"] for r in sub],
                              [r["heavy_apex"] for r in sub])

    sa_b = sa_for("b")
    sa_y = sa_for("y")
    both = [s for s in (sa_b, sa_y) if np.isfinite(s)]
    sa_comb = float(np.mean(both)) if both else float("nan")

    lh = _weighted_pearson([r["light_apex"] for r in F],
                           [r["heavy_apex"] for r in F],
                           [r["pred"] for r in F])

    return {
        "spec_pattern_SA_b": sa_b,
        "spec_pattern_SA_y": sa_y,
        "spec_pattern_SA": sa_comb,
        "spec_pattern_LH_consistency": lh,
        "n_fragments_in_F": len(F),
    }


I2I3J2_KEYS = (
    "pred_hl_ratio_cv",
    "pred_hl_ratio_mad",
    "pred_coverage",
    "pred_coverage_wpred",
    "unexpected_heavy_fraction",
    "unexpected_heavy_intensity_ratio",
)


def _nan_i2i3j2() -> dict:
    return {k: float("nan") for k in I2I3J2_KEYS}


def compute_speclib_i2_i3_j2(frag_records, pred_frags, top_k, seq_len,
                             presence_floor) -> dict:
    """I2 (H/L ratio consistency), I3 (predicted coverage), J2 (unexpected
    heavy on library-unpredicted fragments). Same per-fragment records as I1.
    See spec v1.2 4.4/4.5/4.6. Returns fixed I2I3J2_KEYS; NaN where undefined.
    """
    if not pred_frags or not frag_records:
        logger.debug("i2/i3/j2: no pred_frags or no records -> NaN")
        return _nan_i2i3j2()

    cands, W = [], []
    for r in frag_records:
        fp = frag_pos_for_ion(r["ion_type"], r["ion_num"], seq_len)
        pi = pred_frags.get(frag_key(r["ion_type"], fp, 1))
        if pi is None or not np.isfinite(pi):
            W.append(r)
        else:
            cands.append({**r, "pred": float(pi)})

    out = _nan_i2i3j2()
    if cands:
        cands.sort(key=lambda r: r["pred"], reverse=True)
        F = cands[:top_k]

        logr, wts = [], []
        for r in F:
            if r["light_apex"] > 0 and r["heavy_apex"] > 0:
                logr.append(np.log10(r["heavy_apex"] / r["light_apex"]))
                wts.append(r["pred"])
        if len(logr) >= 2:
            logr = np.asarray(logr, float)
            w = np.asarray(wts, float)
            sw = float(w.sum())
            if sw > 0:
                mean = float(np.sum(w * logr) / sw)
                var = float(np.sum(w * (logr - mean) ** 2) / sw)
                out["pred_hl_ratio_cv"] = float(np.sqrt(max(var, 0.0)))
            med = float(np.median(logr))
            out["pred_hl_ratio_mad"] = float(np.median(np.abs(logr - med)))

        present = [r for r in F if r["heavy_apex"] > presence_floor]
        out["pred_coverage"] = len(present) / len(F)
        sum_pred = sum(r["pred"] for r in F)
        if sum_pred > 0:
            out["pred_coverage_wpred"] = (
                sum(r["pred"] for r in present) / sum_pred)

        if W:
            w_present = [r for r in W if r["heavy_apex"] > presence_floor]
            out["unexpected_heavy_fraction"] = len(w_present) / len(W)
            heavy_F = sum(r["heavy_apex"] for r in F)
            out["unexpected_heavy_intensity_ratio"] = (
                sum(r["heavy_apex"] for r in w_present) / (heavy_F + 1e-9))
    return out
