"""Pure feature math for speclib predicted-intensity features (Phase 1).

No I/O, no pipeline coupling — only numeric vector operations so each piece
is unit-testable in isolation. See
docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md.
"""
import logging

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def spectral_angle(a, b) -> float:
    """Normalized spectral contrast angle similarity in [0, 1].
    1.0 = identical shape (scale-invariant), 0.0 = orthogonal. NaN for
    degenerate input (length < 2, mismatched length, zero norm)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        logger.debug("spectral_angle: degenerate sizes a=%s b=%s", a.size, b.size)
        return float("nan")
    a = np.clip(a, 0.0, None)
    b = np.clip(b, 0.0, None)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        logger.debug("spectral_angle: zero norm na=%g nb=%g", na, nb)
        return float("nan")
    cos = float(np.dot(a, b) / (na * nb))
    cos = min(1.0, max(-1.0, cos))
    if cos > 1.0 - 1e-12:
        cos = 1.0
    elif cos < -1.0 + 1e-12:
        cos = -1.0
    return 1.0 - (2.0 / np.pi) * float(np.arccos(cos))


def spearman_sim(a, b) -> float:
    """Spearman rank correlation in [-1, 1]; robust to absolute-intensity
    miscalibration. NaN for degenerate input."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        logger.debug("spearman_sim: degenerate sizes a=%s b=%s", a.size, b.size)
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        logger.debug("spearman_sim: zero variance")
        return float("nan")
    rho, _ = spearmanr(a, b)
    rho = float(rho)
    if not np.isfinite(rho):
        logger.debug("spearman_sim: non-finite rho")
        return float("nan")
    return rho


def _weighted_pearson(x, y, w) -> float:
    """Pearson correlation of x,y weighted by non-negative weights w.
    NaN for degenerate input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    if x.size < 2 or not (x.size == y.size == w.size):
        logger.debug("_weighted_pearson: degenerate sizes")
        return float("nan")
    sw = float(w.sum())
    if sw < 1e-12:
        logger.debug("_weighted_pearson: zero weight sum")
        return float("nan")
    mx = float(np.sum(w * x) / sw)
    my = float(np.sum(w * y) / sw)
    cov = float(np.sum(w * (x - mx) * (y - my)) / sw)
    vx = float(np.sum(w * (x - mx) ** 2) / sw)
    vy = float(np.sum(w * (y - my) ** 2) / sw)
    if vx < 1e-12 or vy < 1e-12:
        logger.debug("_weighted_pearson: zero weighted variance")
        return float("nan")
    return cov / np.sqrt(vx * vy)


def select_topk_separable(fragments, k):
    """Return the up-to-k separable fragments with the highest predicted
    intensity. `fragments` is a list of dicts with keys 'pred_intensity'
    (float) and 'separable' (bool). Non-separable fragments give no
    light/heavy contrast, so they never occupy a slot."""
    separable = [f for f in fragments if f.get("separable")]
    separable.sort(key=lambda f: f["pred_intensity"], reverse=True)
    chosen = separable[:k]
    logger.debug("select_topk_separable: %d separable, returning %d (k=%d)",
                 len(separable), len(chosen), k)
    return chosen


def i1_pattern_features(pred, obs_heavy, obs_light) -> dict:
    """I1 intensity-pattern consistency (spec §4.3). `pred`, `obs_heavy`,
    `obs_light` are aligned over the same fragment set F (same order). The
    predicted heavy spectrum equals L's predicted intensities placed at the
    heavy fragments (chemical equivalence), so we compare `pred` vs
    `obs_heavy` directly."""
    return {
        "spec_pattern_SA_heavy": spectral_angle(pred, obs_heavy),
        "spec_pattern_spearman_heavy": spearman_sim(pred, obs_heavy),
        "spec_pattern_LH_consistency": _weighted_pearson(obs_light, obs_heavy, pred),
    }
