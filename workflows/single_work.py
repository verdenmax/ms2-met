
import logging
import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # plotting helpers are optional at runtime
    plt = None

from scipy.stats import pearsonr
from numpy import interp

from spectrum.psm_info import PSMInfo
from spectrum.psm_info import HeavyType
from spectrum.psm_info import get_SILAC_increase_mass
from spectrum.psm_info import get_theoretical_isotope_ratios
from spectrum.dia_data import DIAData

from workflows.q1a_helpers import Q1aAccumulator, is_split_window, SHIFT_EPSILON

from constant.keys import ConfigKeys
from configparser import ConfigParser


def multi_batch_work(
    psm1: PSMInfo,
    dia_data1: DIAData,
    psm2: PSMInfo,
    dia_data2: DIAData,
    config: ConfigParser
):
    """ 处理单个肽段，对这单条信息进行处理，计算出是否可信 """

    # logging.info(f"处理信息 {psm}")

    # 从配置中获得 ppm
    mass_tol_ppm = config[ConfigKeys.GENERAL].getfloat(ConfigKeys.MASS_TOL_PPM)
    # 从配置中获得 窗口大小
    xic_cycle_window = config[ConfigKeys.GENERAL].getint(
        ConfigKeys.XIC_CYCLE_WINDOW, fallback=3)

    light_xic = dia_data1.xic_peaks_extreact(
        psm1._rt, xic_cycle_window,
        psm1._precursor_mz, mass_tol_ppm)

    heavy_xic = dia_data2.xic_peaks_extreact(
        psm2._rt, xic_cycle_window,
        psm2._precursor_mz, mass_tol_ppm)

    # 进行画图
    # plot_light_heavy_xic(light_xic, heavy_xic)

    # 计算出 person_corr

    features = {}
    if len(light_xic) == 0 or len(heavy_xic) == 0:
        features["precursor_pearson"] = 0
        features["precursor_apex_delta"] = 0.0
        features["precursor_apex_delta_signed"] = 0.0
        features["precursor_mz_avg_err"] = 0.0
        features["precursor_light_max_int"] = 0.0
        features["precursor_heavy_max_int"] = 0.0
        features["precursor_intensity_ratio"] = 0.0
        features["precursor_cosine"] = 0.0
        features["precursor_snr"] = 0.0
        features["precursor_peak_width_ratio"] = 0.0
        features["precursor_peak_symmetry"] = 0.0
        features["precursor_light_apex_cycle_offset"] = 0
        features["precursor_light_apex_cycle_offset_signed"] = 0
        features["precursor_heavy_apex_cycle_offset"] = 0
        features["precursor_heavy_apex_cycle_offset_signed"] = 0
        features["precursor_base_to_apex_ratio"] = 0.0
        features["precursor_apex_monotonicity"] = 0.0
        features["precursor_n_peaks"] = 0
        features["precursor_smoothness"] = 0.0
    else:
        precursor_score = calc_xic_score(
            light_xic, heavy_xic,
            center_rt=float(psm1._rt),
            heavy_center_rt=float(psm2._rt))
        features["precursor_pearson"] = precursor_score["pearson"]
        features["precursor_apex_delta"] = precursor_score["apex_delta"]
        features["precursor_apex_delta_signed"] = precursor_score["apex_delta_signed"]
        features["precursor_mz_avg_err"] = precursor_score["mz_avg_err"]
        features["precursor_light_max_int"] = precursor_score["light_max_int"]
        features["precursor_heavy_max_int"] = precursor_score["heavy_max_int"]
        features["precursor_intensity_ratio"] = precursor_score["intensity_ratio"]
        features["precursor_cosine"] = precursor_score["cosine"]
        features["precursor_snr"] = precursor_score["snr"]
        features["precursor_peak_width_ratio"] = precursor_score["peak_width_ratio"]
        features["precursor_peak_symmetry"] = precursor_score["peak_symmetry"]
        features["precursor_light_apex_cycle_offset"] = (
            precursor_score["light_apex_cycle_offset"])
        features["precursor_light_apex_cycle_offset_signed"] = (
            precursor_score["light_apex_cycle_offset_signed"])
        features["precursor_heavy_apex_cycle_offset"] = (
            precursor_score["heavy_apex_cycle_offset"])
        features["precursor_heavy_apex_cycle_offset_signed"] = (
            precursor_score["heavy_apex_cycle_offset_signed"])
        features["precursor_base_to_apex_ratio"] = (
            precursor_score["base_to_apex_ratio"])
        features["precursor_apex_monotonicity"] = (
            precursor_score["apex_monotonicity"])
        features["precursor_n_peaks"] = precursor_score["n_peaks"]
        features["precursor_smoothness"] = precursor_score["smoothness"]

    # 同位素模式匹配 + 质量偏移验证
    isotope_spacing = 1.003355 / psm1._charge
    if len(heavy_xic) > 0 and np.max(heavy_xic["intensity"]) > 0:
        apex_idx = np.argmax(heavy_xic["intensity"])
        apex_rt = heavy_xic["rt"][apex_idx]

        heavy_m1_xic = dia_data2.xic_peaks_extreact(
            psm2._rt, xic_cycle_window,
            psm2._precursor_mz + isotope_spacing, mass_tol_ppm)
        heavy_m2_xic = dia_data2.xic_peaks_extreact(
            psm2._rt, xic_cycle_window,
            psm2._precursor_mz + 2 * isotope_spacing, mass_tol_ppm)

        # Sort by rt for np.interp monotonicity (see calc_xic_score).
        if len(heavy_m1_xic) > 0:
            heavy_m1_xic = heavy_m1_xic[np.argsort(heavy_m1_xic["rt"])]
        if len(heavy_m2_xic) > 0:
            heavy_m2_xic = heavy_m2_xic[np.argsort(heavy_m2_xic["rt"])]

        # 在 M0 apex RT 处统一取各同位素峰强度
        m0_int = float(heavy_xic["intensity"][apex_idx])
        m1_int = (float(interp(apex_rt, heavy_m1_xic["rt"],
                                heavy_m1_xic["intensity"]))
                  if len(heavy_m1_xic) > 0 else 0.0)
        m2_int = (float(interp(apex_rt, heavy_m2_xic["rt"],
                                heavy_m2_xic["intensity"]))
                  if len(heavy_m2_xic) > 0 else 0.0)

        obs = np.array([m0_int, m1_int, m2_int])
        theo = np.array(get_theoretical_isotope_ratios(psm1._sequence))
        obs_n = np.linalg.norm(obs)
        theo_n = np.linalg.norm(theo)
        features["isotope_correlation"] = (
            float(np.dot(obs, theo) / (obs_n * theo_n))
            if obs_n > 0 and theo_n > 0 else 0.0)

        features["mass_shift_error"] = float(
            heavy_xic["ppm_error"][apex_idx])
    else:
        features["isotope_correlation"] = 0.0
        features["mass_shift_error"] = 0.0

    pearsons_map = {
        "b": [],
        "y": [],
        "all": [],
    }

    intensitys_map = {
        "b": 0,
        "y": 0,
        "all": 0,
    }

    fragment_apex_deltas = []
    fragment_mz_errs = []
    fragment_intensities = []  # per-ion max intensity for weighted correlation
    fragment_cosines = []
    fragment_snrs = []
    fragment_light_cycle_offsets = []
    fragment_light_cycle_offsets_signed = []
    fragment_heavy_cycle_offsets = []
    fragment_heavy_cycle_offsets_signed = []
    fragment_hl_ratios = {"all": [], "b": [], "y": []}
    fragment_base_to_apex_ratios = []
    fragment_apex_monotonicities = []
    fragment_n_peaks_list = []
    fragment_smoothnesses = []

    # --- Q1a setup: classify co/split-isolation for accumulator ---
    w_light_for_q1a = dia_data1.get_window_info(psm1._precursor_mz)
    heavy_precursor_mz, fragment_ions = psm1.get_heavy_info(HeavyType.SILAC)
    w_heavy_for_q1a = dia_data2.get_window_info(heavy_precursor_mz)
    q1a_acc = Q1aAccumulator(
        split_window=is_split_window(w_light_for_q1a, w_heavy_for_q1a))

    # 枚举所有的信息
    for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:

        # NOTE: 这里应该分情况
        # 如果两个母离子在不同的区间，则均可
        # 如果在相同的区间，并且质量相同，说明重标不影响该碎片离子
        # 说明这个碎片离子不受到重标的影响

        # 计算出 light 信息
        light_ions_xic, light_all_intensity = dia_data1.xic_ms2_peaks_extract(
            psm1._rt, xic_cycle_window,
            precursor_mz=psm1._precursor_mz,
            ions_mass=light_mass,
            mass_tol_ppm=mass_tol_ppm
        )

        # 计算出 heavy 信息
        heavy_ions_xic, heavy_all_intensity = dia_data2.xic_ms2_peaks_extract(
            psm2._rt, xic_cycle_window,
            precursor_mz=psm2._precursor_mz,
            ions_mass=light_mass,
            mass_tol_ppm=mass_tol_ppm
        )

        # --- Q1a: accumulate fragment evidence for SILAC pairing recall ---
        q1a_acc.add(
            ion_type=ions_type,
            light_mass=light_mass, heavy_mass=heavy_mass,
            light_xic=light_ions_xic, heavy_xic=heavy_ions_xic,
        )

        if len(light_ions_xic) == 0 or len(heavy_ions_xic) == 0:
            pearsons_map[ions_type].append(0)
            pearsons_map["all"].append(0)
            fragment_intensities.append(0.0)
            fragment_cosines.append(0.0)
            fragment_snrs.append(0.0)
            continue

        if (np.max(light_ions_xic["intensity"]) > 0 and
                np.max(heavy_ions_xic["intensity"]) > 0):
            intensitys_map[ions_type] += np.sum(light_ions_xic["intensity"])
            intensitys_map[ions_type] += np.sum(heavy_ions_xic["intensity"])
            intensitys_map["all"] += light_all_intensity + \
                heavy_all_intensity

        ion_score = calc_xic_score(
            light_ions_xic, heavy_ions_xic,
            center_rt=float(psm1._rt),
            heavy_center_rt=float(psm2._rt))

        pearsons_map[ions_type].append(ion_score["pearson"])
        pearsons_map["all"].append(ion_score["pearson"])
        fragment_apex_deltas.append(ion_score["apex_delta"])
        fragment_mz_errs.append(ion_score["mz_avg_err"])
        fragment_intensities.append(
            max(ion_score["light_max_int"], ion_score["heavy_max_int"]))
        fragment_cosines.append(ion_score["cosine"])
        fragment_snrs.append(ion_score["snr"])
        fragment_light_cycle_offsets.append(
            ion_score["light_apex_cycle_offset"])
        fragment_light_cycle_offsets_signed.append(
            ion_score["light_apex_cycle_offset_signed"])
        fragment_heavy_cycle_offsets.append(
            ion_score["heavy_apex_cycle_offset"])
        fragment_heavy_cycle_offsets_signed.append(
            ion_score["heavy_apex_cycle_offset_signed"])
        if ion_score["intensity_ratio"] > 0:
            fragment_hl_ratios[ions_type].append(
                float(ion_score["intensity_ratio"]))
            fragment_hl_ratios["all"].append(
                float(ion_score["intensity_ratio"]))
        fragment_base_to_apex_ratios.append(
            ion_score["base_to_apex_ratio"])
        fragment_apex_monotonicities.append(
            ion_score["apex_monotonicity"])
        fragment_n_peaks_list.append(ion_score["n_peaks"])
        fragment_smoothnesses.append(ion_score["smoothness"])

        # logging.info(f"{ions_type} {ions_num} : person({pearson_corr})")

        # plot_light_heavy_xic(light_ions_xic, heavy_ions_xic)

        # rt_values = light_ions_xic["rt"]
        # light_intensitys = light_ions_xic["intensity"]
        # heavy_intensitys = heavy_ions_xic["intensity"]
        #
        # plt.plot(rt_values, light_intensitys, 'o-',
        #          label=f"light_{ions_type} {ions_num}",
        #          linewidth=2, markersize=8)
        # plt.plot(rt_values, heavy_intensitys, 's--',
        #          label=f"light_{ions_type} {ions_num}",
        #          linewidth=2, markersize=8)

    features["valid_fragment_ions_num"] = len(pearsons_map["all"])

    # 分别提取出b离子，y离子，全部的三种特征
    for key, value_list in pearsons_map.items():
        stats = extract_ion_pearson_features(value_list)
        features[f"{key}_count"] = stats["count"]
        features[f"{key}_p25"] = stats["p25"]
        features[f"{key}_p50"] = stats["p50"]
        features[f"{key}_p75"] = stats["p75"]
        features[f"{key}_mean"] = stats["mean"]
        features[f"{key}_std"] = stats["std"]
        features[f"{key}_min"] = stats["min"]
        features[f"{key}_high_ratio"] = stats["high_ratio"]

    # 强度加权碎片相关性
    all_pearsons = pearsons_map["all"]
    total_weight = sum(fragment_intensities)
    if total_weight > 0:
        features["frag_corr_weighted"] = sum(
            p * w for p, w in zip(all_pearsons, fragment_intensities)
        ) / total_weight
    else:
        features["frag_corr_weighted"] = 0.0

    features["matched_intensity_percent"] = (
        (intensitys_map["b"] + intensitys_map["y"]) / intensitys_map["all"] if intensitys_map["all"] > 0 else 0.0)

    # 碎片级 apex_delta / mz_err / cosine / snr 汇总
    features.update(extract_ion_numeric_features(
        fragment_apex_deltas, "all_apex_delta"))
    features.update(extract_ion_numeric_features(
        fragment_mz_errs, "all_mz_err"))
    features.update(extract_ion_numeric_features(
        fragment_cosines, "all_cosine"))
    features.update(extract_ion_numeric_features(
        fragment_snrs, "all_snr"))

    # 碎片级 apex_cycle_offset 汇总（light/heavy × abs/signed × {mean,p50,std,max}）
    features.update(extract_ion_numeric_features(
        fragment_light_cycle_offsets, "all_light_apex_cycle_offset"))
    features.update(extract_ion_numeric_features(
        fragment_light_cycle_offsets_signed,
        "all_light_apex_cycle_offset_signed"))
    features.update(extract_ion_numeric_features(
        fragment_heavy_cycle_offsets, "all_heavy_apex_cycle_offset"))
    features.update(extract_ion_numeric_features(
        fragment_heavy_cycle_offsets_signed,
        "all_heavy_apex_cycle_offset_signed"))

    # H/L 强度比一致性（按 all/b/y 分组的 log10-ratio std/mad）
    for ion_type, ratios in fragment_hl_ratios.items():
        std_v, mad_v = _calc_hl_ratio_consistency(ratios)
        features[f"{ion_type}_log_hl_ratio_std"] = std_v
        features[f"{ion_type}_log_hl_ratio_mad"] = mad_v

    # 碎片级 peak-likeness 汇总（heavy XIC × {mean,p50,std,max}）
    features.update(extract_ion_numeric_features(
        fragment_base_to_apex_ratios, "all_base_to_apex_ratio"))
    features.update(extract_ion_numeric_features(
        fragment_apex_monotonicities, "all_apex_monotonicity"))
    features.update(extract_ion_numeric_features(
        fragment_n_peaks_list, "all_n_peaks"))
    features.update(extract_ion_numeric_features(
        fragment_smoothnesses, "all_smoothness"))

    # 序列级特征
    features["kr_count"] = psm1._sequence.count('K') + \
        psm1._sequence.count('R')
    features["modification_count"] = len(psm1._modify)
    features["total_silac_shift"] = get_SILAC_increase_mass(psm1._sequence)

    # DIA 窗口感知特征
    win_info = dia_data1.get_window_info(psm1._precursor_mz)
    features["window_width"] = win_info["width"]
    features["precursor_centering"] = win_info["centering"]

    # --- Q1a: finalize and merge features ---
    features.update(q1a_acc.compute_features())

    return features


def single_pair_work(
    psm: PSMInfo,
    dia_data: DIAData,
    config: ConfigParser
):
    """ 处理单个肽段，对这单条信息进行处理，计算出是否可信 """

    # if psm._sequence != "ALSSQHQAR":
    #     return None, None
    # TODO:

    # logging.info(f"处理信息 {psm}")

    # 从配置中获得 ppm
    mass_tol_ppm = config[ConfigKeys.GENERAL].getfloat(ConfigKeys.MASS_TOL_PPM)
    # 从配置中获得 窗口大小
    xic_cycle_window = config[ConfigKeys.GENERAL].getint(
        ConfigKeys.XIC_CYCLE_WINDOW, fallback=3)

    light_xic = dia_data.xic_peaks_extreact(
        psm._rt, xic_cycle_window,
        psm._precursor_mz, mass_tol_ppm)

    # --- Q1a setup (single get_heavy_info call shared with fragment loop) ---
    w_light_for_q1a = dia_data.get_window_info(psm._precursor_mz)
    heavy_precursor_mz, fragment_ions = psm.get_heavy_info(HeavyType.SILAC)
    w_heavy_for_q1a = dia_data.get_window_info(heavy_precursor_mz)
    q1a_acc = Q1aAccumulator(
        split_window=is_split_window(w_light_for_q1a, w_heavy_for_q1a))

    heavy_xic = dia_data.xic_peaks_extreact(
        psm._rt, xic_cycle_window,
        heavy_precursor_mz, mass_tol_ppm)

    # 进行画图
    # plot_light_heavy_xic(light_xic, heavy_xic)

    # 计算出 person_corr

    features = {}
    if len(light_xic) == 0 or len(heavy_xic) == 0:
        features["precursor_pearson"] = 0
        features["precursor_apex_delta"] = 0.0
        features["precursor_apex_delta_signed"] = 0.0
        features["precursor_mz_avg_err"] = 0.0
        features["precursor_light_max_int"] = 0.0
        features["precursor_heavy_max_int"] = 0.0
        features["precursor_intensity_ratio"] = 0.0
        features["precursor_cosine"] = 0.0
        features["precursor_snr"] = 0.0
        features["precursor_peak_width_ratio"] = 0.0
        features["precursor_peak_symmetry"] = 0.0
        features["precursor_light_apex_cycle_offset"] = 0
        features["precursor_light_apex_cycle_offset_signed"] = 0
        features["precursor_heavy_apex_cycle_offset"] = 0
        features["precursor_heavy_apex_cycle_offset_signed"] = 0
        features["precursor_base_to_apex_ratio"] = 0.0
        features["precursor_apex_monotonicity"] = 0.0
        features["precursor_n_peaks"] = 0
        features["precursor_smoothness"] = 0.0
    else:
        precursor_score = calc_xic_score(
            light_xic, heavy_xic, center_rt=float(psm._rt))
        features["precursor_pearson"] = precursor_score["pearson"]
        features["precursor_apex_delta"] = precursor_score["apex_delta"]
        features["precursor_apex_delta_signed"] = precursor_score["apex_delta_signed"]
        features["precursor_mz_avg_err"] = precursor_score["mz_avg_err"]
        features["precursor_light_max_int"] = precursor_score["light_max_int"]
        features["precursor_heavy_max_int"] = precursor_score["heavy_max_int"]
        features["precursor_intensity_ratio"] = precursor_score["intensity_ratio"]
        features["precursor_cosine"] = precursor_score["cosine"]
        features["precursor_snr"] = precursor_score["snr"]
        features["precursor_peak_width_ratio"] = precursor_score["peak_width_ratio"]
        features["precursor_peak_symmetry"] = precursor_score["peak_symmetry"]
        features["precursor_light_apex_cycle_offset"] = (
            precursor_score["light_apex_cycle_offset"])
        features["precursor_light_apex_cycle_offset_signed"] = (
            precursor_score["light_apex_cycle_offset_signed"])
        features["precursor_heavy_apex_cycle_offset"] = (
            precursor_score["heavy_apex_cycle_offset"])
        features["precursor_heavy_apex_cycle_offset_signed"] = (
            precursor_score["heavy_apex_cycle_offset_signed"])
        features["precursor_base_to_apex_ratio"] = (
            precursor_score["base_to_apex_ratio"])
        features["precursor_apex_monotonicity"] = (
            precursor_score["apex_monotonicity"])
        features["precursor_n_peaks"] = precursor_score["n_peaks"]
        features["precursor_smoothness"] = precursor_score["smoothness"]

    # 同位素模式匹配 + 质量偏移验证
    isotope_spacing = 1.003355 / psm._charge
    if len(heavy_xic) > 0 and np.max(heavy_xic["intensity"]) > 0:
        apex_idx = np.argmax(heavy_xic["intensity"])
        apex_rt = heavy_xic["rt"][apex_idx]

        heavy_m1_xic = dia_data.xic_peaks_extreact(
            psm._rt, xic_cycle_window,
            heavy_precursor_mz + isotope_spacing, mass_tol_ppm)
        heavy_m2_xic = dia_data.xic_peaks_extreact(
            psm._rt, xic_cycle_window,
            heavy_precursor_mz + 2 * isotope_spacing, mass_tol_ppm)

        # Sort by rt for np.interp monotonicity (see calc_xic_score).
        if len(heavy_m1_xic) > 0:
            heavy_m1_xic = heavy_m1_xic[np.argsort(heavy_m1_xic["rt"])]
        if len(heavy_m2_xic) > 0:
            heavy_m2_xic = heavy_m2_xic[np.argsort(heavy_m2_xic["rt"])]

        # 在 M0 apex RT 处统一取各同位素峰强度
        m0_int = float(heavy_xic["intensity"][apex_idx])
        m1_int = (float(interp(apex_rt, heavy_m1_xic["rt"],
                                heavy_m1_xic["intensity"]))
                  if len(heavy_m1_xic) > 0 else 0.0)
        m2_int = (float(interp(apex_rt, heavy_m2_xic["rt"],
                                heavy_m2_xic["intensity"]))
                  if len(heavy_m2_xic) > 0 else 0.0)

        obs = np.array([m0_int, m1_int, m2_int])
        theo = np.array(get_theoretical_isotope_ratios(psm._sequence))
        obs_n = np.linalg.norm(obs)
        theo_n = np.linalg.norm(theo)
        features["isotope_correlation"] = (
            float(np.dot(obs, theo) / (obs_n * theo_n))
            if obs_n > 0 and theo_n > 0 else 0.0)

        features["mass_shift_error"] = float(
            heavy_xic["ppm_error"][apex_idx])
    else:
        features["isotope_correlation"] = 0.0
        features["mass_shift_error"] = 0.0

    is_same_ms2 = dia_data.check_in_same_ms2(
        psm._precursor_mz, heavy_precursor_mz)

    pearsons_map = {
        "b": [],
        "y": [],
        "all": [],
    }

    intensitys_map = {
        "b": 0,
        "y": 0,
        "all": 0,
    }

    heavy_in_raw = dia_data.check_in_raw(heavy_precursor_mz)

    fragment_apex_deltas = []
    fragment_mz_errs = []
    fragment_intensities = []  # per-ion max intensity for weighted correlation
    fragment_cosines = []
    fragment_snrs = []
    fragment_light_cycle_offsets = []
    fragment_light_cycle_offsets_signed = []
    fragment_heavy_cycle_offsets = []
    fragment_heavy_cycle_offsets_signed = []
    fragment_hl_ratios = {"all": [], "b": [], "y": []}
    fragment_base_to_apex_ratios = []
    fragment_apex_monotonicities = []
    fragment_n_peaks_list = []
    fragment_smoothnesses = []

    ion_data = []  # 存储每个离子的完整数据
    # 枚举所有的信息
    for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:

        if not heavy_in_raw:
            continue

        # 如果在相同的区间，并且质量相同，说明重标不影响该碎片离子
        if np.abs(heavy_mass - light_mass) < SHIFT_EPSILON and is_same_ms2:
            continue

        # 计算出 light 信息
        light_ions_xic, light_all_intensity = dia_data.xic_ms2_peaks_extract(
            psm._rt, xic_cycle_window,
            precursor_mz=psm._precursor_mz,
            ions_mass=light_mass,
            mass_tol_ppm=mass_tol_ppm
        )

        # 计算出 heavy 信息
        heavy_ions_xic, heavy_all_intensity = dia_data.xic_ms2_peaks_extract(
            psm._rt, xic_cycle_window,
            precursor_mz=heavy_precursor_mz,
            ions_mass=heavy_mass,
            mass_tol_ppm=mass_tol_ppm
        )

        q1a_acc.add(
            ion_type=ions_type,
            light_mass=light_mass, heavy_mass=heavy_mass,
            light_xic=light_ions_xic, heavy_xic=heavy_ions_xic,
        )

        if len(light_ions_xic) == 0 or len(heavy_ions_xic) == 0:
            pearsons_map[ions_type].append(0)
            pearsons_map["all"].append(0)
            fragment_intensities.append(0.0)
            fragment_cosines.append(0.0)
            fragment_snrs.append(0.0)
            continue

        if (np.max(light_ions_xic["intensity"]) > 0 and
                np.max(heavy_ions_xic["intensity"]) > 0):
            intensitys_map[ions_type] += np.sum(light_ions_xic["intensity"])
            intensitys_map[ions_type] += np.sum(heavy_ions_xic["intensity"])
            intensitys_map["all"] += light_all_intensity + \
                heavy_all_intensity

        ion_score = calc_xic_score(
            light_ions_xic, heavy_ions_xic, center_rt=float(psm._rt))

        pearsons_map[ions_type].append(ion_score["pearson"])
        pearsons_map["all"].append(ion_score["pearson"])
        fragment_apex_deltas.append(ion_score["apex_delta"])
        fragment_mz_errs.append(ion_score["mz_avg_err"])
        fragment_intensities.append(
            max(ion_score["light_max_int"], ion_score["heavy_max_int"]))
        fragment_cosines.append(ion_score["cosine"])
        fragment_snrs.append(ion_score["snr"])
        fragment_light_cycle_offsets.append(
            ion_score["light_apex_cycle_offset"])
        fragment_light_cycle_offsets_signed.append(
            ion_score["light_apex_cycle_offset_signed"])
        fragment_heavy_cycle_offsets.append(
            ion_score["heavy_apex_cycle_offset"])
        fragment_heavy_cycle_offsets_signed.append(
            ion_score["heavy_apex_cycle_offset_signed"])
        if ion_score["intensity_ratio"] > 0:
            fragment_hl_ratios[ions_type].append(
                float(ion_score["intensity_ratio"]))
            fragment_hl_ratios["all"].append(
                float(ion_score["intensity_ratio"]))
        fragment_base_to_apex_ratios.append(
            ion_score["base_to_apex_ratio"])
        fragment_apex_monotonicities.append(
            ion_score["apex_monotonicity"])
        fragment_n_peaks_list.append(ion_score["n_peaks"])
        fragment_smoothnesses.append(ion_score["smoothness"])

        ion_data.append({
            'ion_type': f"{ions_type}-{ions_num}",
            'ion_num': ions_num,
            'light_mass': light_mass,
            'heavy_mass': heavy_mass,
            'light_rts': light_ions_xic['rt'],
            'light_intensities': light_ions_xic['intensity'],
            'heavy_rts': heavy_ions_xic['rt'],
            'heavy_intensities': heavy_ions_xic['intensity'],
        })

    # plot_light_heavy_contract(ion_data)
    features["valid_fragment_ions_num"] = len(pearsons_map["all"])

    # 分别提取出b离子，y离子，全部的三种特征
    for key, value_list in pearsons_map.items():
        stats = extract_ion_pearson_features(value_list)
        features[f"{key}_count"] = stats["count"]
        features[f"{key}_p25"] = stats["p25"]
        features[f"{key}_p50"] = stats["p50"]
        features[f"{key}_p75"] = stats["p75"]
        features[f"{key}_mean"] = stats["mean"]
        features[f"{key}_std"] = stats["std"]
        features[f"{key}_min"] = stats["min"]
        features[f"{key}_high_ratio"] = stats["high_ratio"]

    # 强度加权碎片相关性
    all_pearsons = pearsons_map["all"]
    total_weight = sum(fragment_intensities)
    if total_weight > 0:
        features["frag_corr_weighted"] = sum(
            p * w for p, w in zip(all_pearsons, fragment_intensities)
        ) / total_weight
    else:
        features["frag_corr_weighted"] = 0.0

    features["matched_intensity_percent"] = (
        (intensitys_map["b"] + intensitys_map["y"]) / intensitys_map["all"] if intensitys_map["all"] > 0 else 0.0)

    # 碎片级 apex_delta / mz_err / cosine / snr 汇总
    features.update(extract_ion_numeric_features(
        fragment_apex_deltas, "all_apex_delta"))
    features.update(extract_ion_numeric_features(
        fragment_mz_errs, "all_mz_err"))
    features.update(extract_ion_numeric_features(
        fragment_cosines, "all_cosine"))
    features.update(extract_ion_numeric_features(
        fragment_snrs, "all_snr"))

    # 碎片级 apex_cycle_offset 汇总（light/heavy × abs/signed × {mean,p50,std,max}）
    features.update(extract_ion_numeric_features(
        fragment_light_cycle_offsets, "all_light_apex_cycle_offset"))
    features.update(extract_ion_numeric_features(
        fragment_light_cycle_offsets_signed,
        "all_light_apex_cycle_offset_signed"))
    features.update(extract_ion_numeric_features(
        fragment_heavy_cycle_offsets, "all_heavy_apex_cycle_offset"))
    features.update(extract_ion_numeric_features(
        fragment_heavy_cycle_offsets_signed,
        "all_heavy_apex_cycle_offset_signed"))

    # H/L 强度比一致性（按 all/b/y 分组的 log10-ratio std/mad）
    for ion_type, ratios in fragment_hl_ratios.items():
        std_v, mad_v = _calc_hl_ratio_consistency(ratios)
        features[f"{ion_type}_log_hl_ratio_std"] = std_v
        features[f"{ion_type}_log_hl_ratio_mad"] = mad_v

    # 碎片级 peak-likeness 汇总（heavy XIC × {mean,p50,std,max}）
    features.update(extract_ion_numeric_features(
        fragment_base_to_apex_ratios, "all_base_to_apex_ratio"))
    features.update(extract_ion_numeric_features(
        fragment_apex_monotonicities, "all_apex_monotonicity"))
    features.update(extract_ion_numeric_features(
        fragment_n_peaks_list, "all_n_peaks"))
    features.update(extract_ion_numeric_features(
        fragment_smoothnesses, "all_smoothness"))

    # 序列级特征
    features["kr_count"] = psm._sequence.count('K') + \
        psm._sequence.count('R')
    features["modification_count"] = len(psm._modify)
    features["total_silac_shift"] = get_SILAC_increase_mass(psm._sequence)

    if heavy_in_raw:
        features["heavy_in_raw"] = 1
    else:
        features["heavy_in_raw"] = 0

    # DIA 窗口感知特征
    win_info = dia_data.get_window_info(psm._precursor_mz)
    features["window_width"] = win_info["width"]
    features["precursor_centering"] = win_info["centering"]

    features.update(q1a_acc.compute_features())

    return features


def plot_light_heavy_contract(ion_data):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同的离子类型分配颜色
    ion_types = list(set([data['ion_type'] for data in ion_data]))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']
    color_map = {ion_type: colors[i % len(colors)]
                 for i, ion_type in enumerate(ion_types)}

    # 绘制每个离子的XIC线
    for data in ion_data:
        ion_type = data['ion_type']
        color = color_map[ion_type]

        # 绘制轻标离子的XIC线
        if len(data['light_rts']) > 0:
            # 对RT进行排序，确保线是按顺序连接的
            sorted_indices = np.argsort(data['light_rts'])
            sorted_light_rts = data['light_rts'][sorted_indices]
            sorted_light_intensities = data['light_intensities'][sorted_indices]

            # 创建轻标点的3D坐标
            light_points = np.column_stack([
                sorted_light_rts,
                np.full_like(sorted_light_rts, data['light_mass']),
                sorted_light_intensities
            ])

            # 绘制轻标XIC线
            ax.plot(
                light_points[:, 0],
                light_points[:, 1],
                light_points[:, 2],
                color=color,
                linewidth=2,
                alpha=0.7,
                label=f'{ion_type} Light'
            )

            # 在XIC线的峰值点标注离子类型
            max_intensity_idx = np.argmax(sorted_light_intensities)
            ax.text(
                light_points[max_intensity_idx, 0],
                light_points[max_intensity_idx, 1],
                light_points[max_intensity_idx, 2],
                f'{ion_type}-L',
                fontsize=9,
                color=color,
                fontweight='bold'
            )

        # 绘制重标离子的XIC线（使用虚线）
        if len(data['heavy_rts']) > 0:
            # 对RT进行排序，确保线是按顺序连接的
            sorted_indices = np.argsort(data['heavy_rts'])
            sorted_heavy_rts = data['heavy_rts'][sorted_indices]
            sorted_heavy_intensities = data['heavy_intensities'][sorted_indices]

            # 创建重标点的3D坐标
            heavy_points = np.column_stack([
                sorted_heavy_rts,
                np.full_like(sorted_heavy_rts, data['heavy_mass']),
                sorted_heavy_intensities
            ])

            # 绘制重标XIC线（使用虚线）
            ax.plot(
                heavy_points[:, 0],
                heavy_points[:, 1],
                heavy_points[:, 2],
                color=color,
                linestyle='--',
                linewidth=2,
                alpha=0.7,
                label=f'{ion_type} Heavy'
            )

            # 在XIC线的峰值点标注离子类型
            max_intensity_idx = np.argmax(sorted_heavy_intensities)
            ax.text(
                heavy_points[max_intensity_idx, 0],
                heavy_points[max_intensity_idx, 1],
                heavy_points[max_intensity_idx, 2],
                f'{ion_type}-H',
                fontsize=9,
                color=color,
                fontweight='bold',
                style='italic'
            )

    # 设置标签
    ax.set_xlabel('Retention Time (RT)', fontsize=12, labelpad=10)
    ax.set_ylabel('m/z', fontsize=12, labelpad=10)
    ax.set_zlabel('Intensity', fontsize=12, labelpad=10)
    ax.set_title(
        '3D Fragment Ions XIC: Light (solid) vs Heavy (dashed)',
        fontsize=14,
        pad=20
    )

    # 添加图例，但避免重复
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper left', bbox_to_anchor=(0, 0.9))

    # 调整视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


def extract_ion_pearson_features(ions_pearsons: []) -> dict:
    """
    计算出这个数组中的分位数、均值、标准差、最小值、高相关占比
    """
    clean_vals = [v for v in ions_pearsons if not np.isnan(
        v) and np.isfinite(v)]
    count = len(clean_vals)

    if count == 0:
        return {
            "count": 0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "mean": 0,
            "std": 0,
            "min": 0,
            "high_ratio": 0,
        }

    p25 = np.clip(np.percentile(clean_vals, 25), 0, 1)
    p50 = np.clip(np.percentile(clean_vals, 50), 0, 1)
    p75 = np.clip(np.percentile(clean_vals, 75), 0, 1)
    mean = np.mean(clean_vals)
    # Bug #21: N=1 has no defined spread. Returning 0 conflates a single
    # ion with many ions of identical value. Use NaN so HistGBT can
    # branch on missingness.
    std = float(np.std(clean_vals)) if count >= 2 else float("nan")
    min_val = float(np.min(clean_vals))
    high_ratio = sum(1 for v in clean_vals if v > 0.5) / count

    return {
        "count": count,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "mean": mean,
        "std": std,
        "min": min_val,
        "high_ratio": high_ratio,
    }


def extract_ion_numeric_features(values: list, prefix: str) -> dict:
    """
    对碎片级数值列表（如 apex_delta、mz_err、cycle_offset）计算均值、
    中位数、标准差和最大值。清除 NaN/Inf 值后统计。
    """
    clean_vals = [v for v in values if not np.isnan(v) and np.isfinite(v)]
    if len(clean_vals) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(clean_vals)),
        f"{prefix}_p50": float(np.median(clean_vals)),
        f"{prefix}_std": float(np.std(clean_vals)),
        f"{prefix}_max": float(np.max(clean_vals)),
    }


def _calc_fwhm(rt: np.ndarray, intensity: np.ndarray) -> float:
    """计算 XIC 的半高全宽 (FWHM)，返回 RT 单位的宽度"""
    if len(intensity) < 3:
        return 0.0
    max_int = float(np.nanmax(intensity))
    if max_int <= 0 or np.isnan(max_int):
        return 0.0
    half_max = max_int * 0.5
    above = intensity >= half_max
    indices = np.where(above)[0]
    if len(indices) < 2:
        return 0.0
    return float(rt[indices[-1]] - rt[indices[0]])


def _calc_symmetry(intensity: np.ndarray) -> float:
    """计算峰对称性: |左半面积 - 右半面积| / 总面积，越接近 0 越对称"""
    if len(intensity) < 3:
        return 0.0
    total = float(np.nansum(intensity))
    if total <= 0 or np.isnan(total):
        return 0.0
    apex_idx = np.nanargmax(intensity)
    left_area = float(np.nansum(intensity[:apex_idx + 1]))
    right_area = float(np.nansum(intensity[apex_idx + 1:]))
    return abs(left_area - right_area) / total


def _calc_snr(intensity: np.ndarray) -> float:
    """Signal-to-noise ratio using p25 as a robust noise floor.

    For sparse SILAC peaks (typical: 1-2 nonzero scans out of 7),
    nanmedian is exactly 0 and using max/median blows up. Instead,
    use 25th-percentile of all values as the noise estimate, and
    floor it by max_int * 1e-3 to bound the ratio at 1000.
    """
    intensity = np.asarray(intensity, dtype="f8")
    if intensity.size == 0:
        return 0.0
    max_int = float(np.nanmax(intensity))
    if not np.isfinite(max_int) or max_int <= 0:
        return 0.0
    p25 = float(np.percentile(intensity, 25))
    noise_floor = max_int * 1e-3  # cap SNR at 1000
    noise = max(p25, noise_floor)
    return max_int / noise


def _calc_base_to_apex_ratio(intensity: np.ndarray) -> float:
    """Edge average / apex intensity.

    True peaks: edges decay to near 0 -> ratio close to 0.
    Plateau / background / multi-peak stacks -> ratio close to 1.

    Returns 0.0 for empty / short / all-zero XIC, or if any value is
    non-finite (NaN would silently propagate through np.max and produce
    nan/nan, polluting downstream features).
    """
    if len(intensity) < 3:
        return 0.0
    if not np.all(np.isfinite(intensity)):
        return 0.0
    apex = float(np.max(intensity))
    if apex <= 0:
        return 0.0
    base = (float(intensity[0]) + float(intensity[-1])) / 2
    return base / apex


def _calc_apex_monotonicity(intensity: np.ndarray) -> float:
    """Fraction of pairs that monotonically rise to apex and fall after.

    Left of apex should be non-decreasing; right of apex should be
    non-increasing. Return = 1 - (violations / total_pairs) in [0, 1].
    True peaks -> ~1; zigzag / noise -> low.

    Returns 0.0 for empty / short XIC, or if any value is non-finite
    (NaN would silently inflate the score because nan compares False
    against any number, so np.diff(...) < 0 yields False at NaN gaps).

    Note: right slice includes apex (intensity[apex_idx:]) so when apex
    is at the leftmost index there is still a meaningful right slice.
    """
    if len(intensity) < 3:
        return 0.0
    if not np.all(np.isfinite(intensity)):
        return 0.0
    apex_idx = int(np.argmax(intensity))
    left = intensity[:apex_idx + 1]
    right = intensity[apex_idx:]
    if len(left) < 2 and len(right) < 2:
        return 0.0
    left_viol = int(np.sum(np.diff(left) < 0)) if len(left) >= 2 else 0
    right_viol = int(np.sum(np.diff(right) > 0)) if len(right) >= 2 else 0
    total_pairs = max(len(intensity) - 1, 1)
    return 1.0 - (left_viol + right_viol) / total_pairs


def _calc_n_peaks(
    intensity: np.ndarray, prominence_frac: float = 0.3
) -> int:
    """Count local maxima with prominence >= prominence_frac * apex.

    True chromatographic peak -> 1; co-elution / interference -> 2+.
    The prominence threshold filters out small bumps that are likely
    noise rather than separate peaks. Endpoints are not counted.

    Returns 0 for empty / short XIC, all-zero XIC, or non-finite input
    (scipy.signal.find_peaks behavior on NaN is undefined).
    """
    if len(intensity) < 3:
        return 0
    if not np.all(np.isfinite(intensity)):
        return 0
    from scipy.signal import find_peaks
    max_int = float(np.max(intensity))
    if max_int <= 0:
        return 0
    peaks, _ = find_peaks(intensity, prominence=max_int * prominence_frac)
    return int(len(peaks))


def _calc_smoothness(intensity: np.ndarray) -> float:
    """Sum of squared second differences / total^2.

    Smooth Gaussian-like peaks -> close to 0.
    Sharp zigzag / single-point spikes -> large value.
    Normalized by total^2 to make cross-sample comparable; note this
    is NOT normalized by length, so different xic_cycle_window settings
    produce different absolute values.
    """
    if len(intensity) < 3:
        return 0.0
    if not np.all(np.isfinite(intensity)):
        return 0.0
    total = float(np.sum(intensity))
    if total <= 0:
        return 0.0
    second_diff = np.diff(intensity, n=2)
    return float(np.sum(second_diff ** 2) / (total ** 2 + 1e-12))


def _calc_cycle_offset(xic: np.ndarray, center_rt: float) -> tuple[int, int]:
    """Compute how far the intensity apex is from the center RT, in cycles.

    Returns (abs_offset, signed_offset). signed < 0 means apex is at an
    earlier cycle than center_rt; > 0 means later.

    The "center" is the cycle whose RT is closest to center_rt (among
    entries with valid cycle_idx >= 0). The "apex" is the cycle at
    argmax(intensity). Both returned values are integer cycle counts.

    Returns (0, 0) for empty XIC, all-invalid cycle_idx, all-zero
    intensity (no real apex — np.argmax would pick the window edge),
    or apex with cycle_idx == -1 (defensive).
    """
    if len(xic) == 0:
        return 0, 0
    valid_mask = xic["cycle_idx"] >= 0
    if not np.any(valid_mask):
        return 0, 0
    if not np.any(xic["intensity"] > 0):
        return 0, 0
    valid_xic = xic[valid_mask]
    center_local_idx = int(np.argmin(np.abs(valid_xic["rt"] - center_rt)))
    center_cycle = int(valid_xic["cycle_idx"][center_local_idx])
    apex_global_idx = int(np.argmax(xic["intensity"]))
    apex_cycle = int(xic["cycle_idx"][apex_global_idx])
    if apex_cycle < 0:
        return 0, 0
    signed = apex_cycle - center_cycle
    return abs(signed), signed


def _calc_hl_ratio_consistency(ratios: list) -> tuple[float, float]:
    """Compute consistency of light/heavy intensity ratios across fragments.

    Returns (std, mad) of log10(ratio) over the input list. Non-positive
    ratios are dropped (cannot take log). std uses NaN for count==1 to
    match the existing single-element convention (see Bug #21 in
    extract_ion_pearson_features). mad is 0 for empty input, otherwise
    median absolute deviation from the median.
    """
    log_ratios = [float(np.log10(r)) for r in ratios if r > 0]
    count = len(log_ratios)
    if count == 0:
        return 0.0, 0.0
    if count == 1:
        return float("nan"), 0.0
    arr = np.asarray(log_ratios, dtype="f8")
    std_v = float(np.std(arr))
    med = float(np.median(arr))
    mad_v = float(np.median(np.abs(arr - med)))
    return std_v, mad_v


def _default_xic_score() -> dict:
    """calc_xic_score 的全零默认返回值"""
    return {
        "pearson": np.float32(0.0),
        "mz_avg_err": 0.0,
        "apex_delta": 0.0,
        "apex_delta_signed": 0.0,
        "light_max_int": 0.0,
        "heavy_max_int": 0.0,
        "intensity_ratio": 0.0,
        "cosine": 0.0,
        "snr": 0.0,
        "peak_width_ratio": 0.0,
        "peak_symmetry": 0.0,
        "light_apex_cycle_offset": 0,
        "light_apex_cycle_offset_signed": 0,
        "heavy_apex_cycle_offset": 0,
        "heavy_apex_cycle_offset_signed": 0,
        "base_to_apex_ratio": 0.0,
        "apex_monotonicity": 0.0,
        "n_peaks": 0,
        "smoothness": 0.0,
    }


def calc_xic_score(
    light_xic: np.array, heavy_xic: np.array,
    center_rt: float | None = None,
    heavy_center_rt: float | None = None,
    intensity_threshold: float = 1e-10
) -> dict:
    """ 根据轻重标 XIC 计算综合特征，返回包含 pearson/mz_avg_err/apex_delta/强度信息的字典 """

    # Defensively sort by rt — np.interp requires monotonically increasing xp;
    # raw mzML scan order may not guarantee that for multiplexed DIA.
    if len(light_xic) > 0:
        light_xic = light_xic[np.argsort(light_xic["rt"])]
    if len(heavy_xic) > 0:
        heavy_xic = heavy_xic[np.argsort(heavy_xic["rt"])]

    # 计算重标平均误差
    ppm_errors = heavy_xic["ppm_error"]
    if np.all(np.isnan(ppm_errors)):
        mz_avg_err = 0.0
    else:
        mz_avg_err = float(np.nanmean(ppm_errors))

    # 计算峰顶的时间差
    rt_apex_light = light_xic["rt"][np.argmax(light_xic["intensity"])]
    rt_apex_heavy = heavy_xic["rt"][np.argmax(heavy_xic["intensity"])]
    apex_delta = float(abs(rt_apex_light - rt_apex_heavy))
    # Bug #19: also emit the signed delta. The negative-sample generator
    # always shifts heavy by +10, so without sign the model can't tell
    # "heavy elutes earlier" (real co-elution variation) from the
    # artificial direction of the shift used for negatives.
    apex_delta_signed = float(rt_apex_light - rt_apex_heavy)

    # 计算强度信息
    light_max_int = float(np.max(light_xic["intensity"]))
    heavy_max_int = float(np.max(heavy_xic["intensity"]))
    light_total = float(np.sum(light_xic["intensity"]))
    heavy_total = float(np.sum(heavy_xic["intensity"]))
    intensity_ratio = light_total / heavy_total if heavy_total > 0 else 0.0

    # 峰形特征（在原始 XIC 上计算，不依赖插值）
    snr = _calc_snr(heavy_xic["intensity"])
    peak_symmetry = _calc_symmetry(heavy_xic["intensity"])
    light_fwhm = _calc_fwhm(light_xic["rt"], light_xic["intensity"])
    heavy_fwhm = _calc_fwhm(heavy_xic["rt"], heavy_xic["intensity"])
    peak_width_ratio = (heavy_fwhm / light_fwhm
                        if light_fwhm > 0 else 0.0)
    base_to_apex_ratio = _calc_base_to_apex_ratio(heavy_xic["intensity"])
    apex_monotonicity = _calc_apex_monotonicity(heavy_xic["intensity"])
    n_peaks = _calc_n_peaks(heavy_xic["intensity"])
    smoothness = _calc_smoothness(heavy_xic["intensity"])

    # 计算峰相关性
    # 统一时间轴
    rt_start = max(light_xic["rt"].min(), heavy_xic["rt"].min())
    rt_end = min(light_xic["rt"].max(), heavy_xic["rt"].max())

    if rt_start >= rt_end:
        result = _default_xic_score()
        result["mz_avg_err"] = mz_avg_err
        result["apex_delta"] = apex_delta
        result["apex_delta_signed"] = apex_delta_signed
        result["light_max_int"] = light_max_int
        result["heavy_max_int"] = heavy_max_int
        result["intensity_ratio"] = intensity_ratio
        result["snr"] = snr
        result["peak_width_ratio"] = peak_width_ratio
        result["peak_symmetry"] = peak_symmetry
        result["base_to_apex_ratio"] = base_to_apex_ratio
        result["apex_monotonicity"] = apex_monotonicity
        result["n_peaks"] = n_peaks
        result["smoothness"] = smoothness
        if center_rt is not None:
            l_abs, l_sig = _calc_cycle_offset(light_xic, center_rt)
            h_center = (heavy_center_rt
                        if heavy_center_rt is not None else center_rt)
            h_abs, h_sig = _calc_cycle_offset(heavy_xic, h_center)
            result["light_apex_cycle_offset"] = l_abs
            result["light_apex_cycle_offset_signed"] = l_sig
            result["heavy_apex_cycle_offset"] = h_abs
            result["heavy_apex_cycle_offset_signed"] = h_sig
        return result

    common_rt = np.linspace(rt_start, rt_end, 100)
    inten1_interp = interp(common_rt, light_xic["rt"], light_xic["intensity"])
    inten2_interp = interp(common_rt, heavy_xic["rt"], heavy_xic["intensity"])

    # 检查是否都是0或接近0
    light_near_zero = np.all(np.abs(inten1_interp) < intensity_threshold)
    heavy_near_zero = np.all(np.abs(inten2_interp) < intensity_threshold)

    if light_near_zero and heavy_near_zero:
        corr = 0.0
        cosine = 0.0
    elif light_near_zero or heavy_near_zero:
        corr = 0.0
        cosine = 0.0
    else:
        if np.std(inten1_interp) < 1e-10 or np.std(inten2_interp) < 1e-10:
            corr = 0.0
        else:
            try:
                corr, _ = pearsonr(inten1_interp, inten2_interp)
            except (ValueError, RuntimeWarning):
                corr = 0.0
            # Bug #22: modern scipy emits ConstantInputWarning and returns
            # NaN (no longer raises) when one input is constant. Coerce
            # NaN/Inf pearson → 0.0 explicitly so downstream features
            # don't silently inherit NaN.
            if not np.isfinite(corr):
                corr = 0.0
        # cosine similarity
        norm1 = np.linalg.norm(inten1_interp)
        norm2 = np.linalg.norm(inten2_interp)
        if norm1 > 0 and norm2 > 0:
            cosine = float(np.dot(inten1_interp, inten2_interp)
                           / (norm1 * norm2))
        else:
            cosine = 0.0

    if center_rt is not None:
        l_abs, l_sig = _calc_cycle_offset(light_xic, center_rt)
        h_center = (heavy_center_rt
                    if heavy_center_rt is not None else center_rt)
        h_abs, h_sig = _calc_cycle_offset(heavy_xic, h_center)
    else:
        l_abs = l_sig = h_abs = h_sig = 0

    return {
        "pearson": np.float32(corr),
        "mz_avg_err": mz_avg_err,
        "apex_delta": apex_delta,
        "apex_delta_signed": apex_delta_signed,
        "light_max_int": light_max_int,
        "heavy_max_int": heavy_max_int,
        "intensity_ratio": intensity_ratio,
        "cosine": cosine,
        "snr": snr,
        "peak_width_ratio": peak_width_ratio,
        "peak_symmetry": peak_symmetry,
        "light_apex_cycle_offset": l_abs,
        "light_apex_cycle_offset_signed": l_sig,
        "heavy_apex_cycle_offset": h_abs,
        "heavy_apex_cycle_offset_signed": h_sig,
        "base_to_apex_ratio": base_to_apex_ratio,
        "apex_monotonicity": apex_monotonicity,
        "n_peaks": n_peaks,
        "smoothness": smoothness,
    }


def plot_light_heavy_xic(light_xic, heavy_xic):
    """ 画图 """
    rt_values = light_xic["rt"]
    light_intensitys = light_xic["intensity"]
    heavy_intensitys = heavy_xic["intensity"]

    # 创建画布和坐标轴
    plt.figure(figsize=(10, 6))
    # 绘制两条折线
    plt.plot(rt_values, light_intensitys, 'o-',
             label='light', linewidth=2, markersize=8)
    plt.plot(rt_values, heavy_intensitys, 's--',
             label='heavy', linewidth=2, markersize=8)

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 自动调整布局并显示
    plt.tight_layout()
    plt.show()
