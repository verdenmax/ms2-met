
import numpy as np


def match_peak_ppm(
    mz_arr: np.ndarray, intensity_arr: np.ndarray,
    precursor_mz: np.float32, mass_tol_ppm: np.float32
) -> (np.float32, np.float32):
    """  进行峰的匹配，最后返回ppm 误差和 intensity """
    # 计算每个峰的 ppm 误差
    ppm_errors = (mz_arr - precursor_mz) / precursor_mz * 1e6

    # 找出在容差范围内的索引
    mask = np.abs(ppm_errors) <= mass_tol_ppm

    if not np.any(mask):
        # 没有匹配
        return np.float32(np.nan), np.float32(0.0)

    # 获取容差范围内的所有数据
    matched_ppm_errors = ppm_errors[mask]
    matched_intensities = intensity_arr[mask]

    # 计算误差范围内所有峰的强度之和
    total_intensity = np.sum(matched_intensities)

    # A tolerance window may contain more than one centroid.  Report the
    # centroid of the matched ion cluster, not the unweighted mean of peak
    # positions (which lets tiny noise peaks move the error arbitrarily).
    if total_intensity > 0:
        ppm_error = np.average(
            matched_ppm_errors, weights=matched_intensities)
    else:
        ppm_error = np.average(matched_ppm_errors)

    return np.float32(ppm_error), np.float32(total_intensity)


def match_peak_panel_ppm(
    mz_arr: np.ndarray,
    intensity_arr: np.ndarray,
    target_mz: np.ndarray,
    mass_tol_ppm: np.float32,
) -> tuple[np.float32, np.float32]:
    """Match a target panel while counting every observed centroid once.

    Natural-isotope exact masses can be closer than two ppm windows. A naïve
    sum of independent ``match_peak_ppm`` calls double-counts centroids in
    overlapping windows. This union matcher assigns each matched centroid to
    its closest theoretical target for ppm reporting and sums it once.
    """
    targets = np.asarray(target_mz, dtype="f8")
    if targets.ndim != 1 or not len(targets):
        raise ValueError("target_mz must be a non-empty one-dimensional panel")
    if not np.isfinite(targets).all() or (targets <= 0).any():
        raise ValueError("target_mz must contain positive finite values")
    mz_values = np.asarray(mz_arr, dtype="f8")
    intensities = np.asarray(intensity_arr)
    if len(mz_values) != len(intensities):
        raise ValueError("mz_arr and intensity_arr must have equal lengths")
    if not len(mz_values):
        return np.float32(np.nan), np.float32(0.0)
    ppm = (mz_values[None, :] - targets[:, None]) \
        / targets[:, None] * 1e6
    absolute = np.abs(ppm)
    matched = np.any(absolute <= mass_tol_ppm, axis=0)
    if not matched.any():
        return np.float32(np.nan), np.float32(0.0)
    matched_intensities = intensities[matched]
    closest_target = np.argmin(absolute[:, matched], axis=0)
    columns = np.arange(int(np.count_nonzero(matched)))
    closest_ppm = ppm[:, matched][closest_target, columns]
    total_intensity = float(np.sum(matched_intensities))
    ppm_error = (
        float(np.average(closest_ppm, weights=matched_intensities))
        if total_intensity > 0 else float(np.mean(closest_ppm))
    )
    return np.float32(ppm_error), np.float32(total_intensity)


def match_peak_targets_ppm(
    mz_arr: np.ndarray,
    intensity_arr: np.ndarray,
    target_mz: np.ndarray,
    mass_tol_ppm: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Match many independent targets in one sorted centroid spectrum.

    Unlike :func:`match_peak_panel_ppm`, targets remain independent: an
    observed centroid may contribute to two overlapping fragment windows.
    Searchsorted bounds avoid allocating a ``targets x peaks`` matrix and let
    a caller load each spectrum only once for a complete fragment panel.
    """
    targets = np.asarray(target_mz, dtype="f8")
    mz_values = np.asarray(mz_arr, dtype="f8")
    intensities = np.asarray(intensity_arr)
    if targets.ndim != 1:
        raise ValueError("target_mz must be one-dimensional")
    if not np.isfinite(targets).all() or (targets <= 0).any():
        raise ValueError("target_mz must contain positive finite values")
    if len(mz_values) != len(intensities):
        raise ValueError("mz_arr and intensity_arr must have equal lengths")
    errors = np.full(len(targets), np.nan, dtype="f4")
    matched = np.zeros(len(targets), dtype="f4")
    if not len(targets) or not len(mz_values):
        return errors, matched
    if len(mz_values) > 1 and np.any(mz_values[1:] < mz_values[:-1]):
        order = np.argsort(mz_values, kind="stable")
        mz_values = mz_values[order]
        intensities = intensities[order]

    tolerance = float(mass_tol_ppm) * 1e-6
    lower = targets * (1.0 - tolerance)
    upper = targets * (1.0 + tolerance)
    starts = np.searchsorted(mz_values, lower, side="left")
    stops = np.searchsorted(mz_values, upper, side="right")
    for index, (start, stop) in enumerate(zip(starts, stops)):
        if start == stop:
            continue
        values = intensities[start:stop]
        ppm = (mz_values[start:stop] - targets[index]) \
            / targets[index] * 1e6
        total = float(np.sum(values))
        matched[index] = total
        errors[index] = (
            np.average(ppm, weights=values)
            if total > 0 else np.mean(ppm)
        )
    return errors, matched


def centroid_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    rel_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid a single profile-mode spectrum.

    Picks local-maxima with `intensity[i] >= max(intensity) * rel_threshold`,
    then refines the m/z location by 3-point parabolic interpolation.
    Intensity is reported as the apex sample height. Output dtype matches
    input.

    Args:
        mz: 1D array of m/z values, assumed monotonically increasing.
        intensity: 1D array of intensities, same length as `mz`.
        rel_threshold: drop maxima with intensity below
            `max(intensity) * rel_threshold`. Default 1e-3.

    Plateaus with equal adjacent samples are resolved to the leftmost
    interior index (asymmetric ``>`` / ``>=`` rule).

    Returns:
        (mz_out, intensity_out): two 1D arrays of equal length (= number of
        accepted peaks). Empty arrays when input length < 3 or no peak
        survives the threshold.
    """
    n = len(mz)
    if len(intensity) != n:
        raise ValueError(
            f"mz and intensity must have the same length; "
            f"got len(mz)={n}, len(intensity)={len(intensity)}")
    if n < 3:
        empty_mz = np.empty(0, dtype=mz.dtype if n > 0 else np.float32)
        empty_int = np.empty(
            0, dtype=intensity.dtype if n > 0 else np.float32)
        return empty_mz, empty_int

    interior = intensity[1:-1]
    left = intensity[:-2]
    right = intensity[2:]
    is_peak = (interior > left) & (interior >= right)
    peak_idx = np.where(is_peak)[0] + 1
    if peak_idx.size == 0:
        return (np.empty(0, dtype=mz.dtype),
                np.empty(0, dtype=intensity.dtype))

    max_intensity = float(intensity.max())
    if max_intensity <= 0.0:
        return (np.empty(0, dtype=mz.dtype),
                np.empty(0, dtype=intensity.dtype))
    cutoff = max_intensity * rel_threshold
    peak_idx = peak_idx[intensity[peak_idx] >= cutoff]
    if peak_idx.size == 0:
        return (np.empty(0, dtype=mz.dtype),
                np.empty(0, dtype=intensity.dtype))

    y0 = intensity[peak_idx - 1].astype(np.float64)
    y1 = intensity[peak_idx].astype(np.float64)
    y2 = intensity[peak_idx + 1].astype(np.float64)

    denom = (y0 - 2.0 * y1 + y2)
    safe = np.abs(denom) > 1e-12
    dx = np.zeros_like(denom)
    np.divide(0.5 * (y0 - y2), denom, out=dx, where=safe)
    # Theoretical bound for a true parabolic max is |dx| <= 0.5. A larger
    # offset signals a degenerate fit (noisy plateau adjacent to a steep
    # edge, or numerical edge-case) — treat as "fit failed" and fall back
    # to bin-center (dx = 0).
    dx[np.abs(dx) > 0.5] = 0.0

    mz_center = mz[peak_idx].astype(np.float64)
    mz_prev = mz[peak_idx - 1].astype(np.float64)
    mz_next = mz[peak_idx + 1].astype(np.float64)
    half_step = (mz_next - mz_prev) * 0.5
    refined_mz = mz_center + dx * half_step

    out_mz = refined_mz.astype(mz.dtype)
    out_int = y1.astype(intensity.dtype)
    return out_mz, out_int
