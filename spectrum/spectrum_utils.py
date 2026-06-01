
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

    # 找出强度最大的峰的索引（用于返回其ppm误差）
    ppm_error = np.average(matched_ppm_errors)

    return np.float32(ppm_error), np.float32(total_intensity)


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
