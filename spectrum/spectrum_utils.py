
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
