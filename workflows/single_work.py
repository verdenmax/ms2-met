
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from numpy import interp

from spectrum.psm_info import PSMInfo
from spectrum.psm_info import HeavyType
from spectrum.dia_data import DIAData

from constant.keys import ConfigKeys
from configparser import ConfigParser


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

    heavy_precursor_mz, fragment_ions = psm.get_heavy_info(HeavyType.SILAC)

    heavy_xic = dia_data.xic_peaks_extreact(
        psm._rt, xic_cycle_window,
        heavy_precursor_mz, mass_tol_ppm)

    # 进行画图
    # plot_light_heavy_xic(light_xic, heavy_xic)

    # 计算出 person_corr

    features = {}
    if len(light_xic) == 0 or len(heavy_xic) == 0:
        features["precursor_pearson"] = 0
    else:
        person_corr = calc_xic_score(light_xic, heavy_xic)
        features["precursor_pearson"] = person_corr

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
        "all": 1,
    }
    # 枚举所有的信息
    for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:

        # NOTE: 这里应该分情况
        # 如果两个母离子在不同的区间，则均可
        # 如果在相同的区间，并且质量相同，说明重标不影响该碎片离子
        # 说明这个碎片离子不受到重标的影响
        if np.abs(heavy_mass - light_mass) < 0.01 and is_same_ms2:
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

        if len(light_ions_xic) == 0 or len(heavy_ions_xic) == 0:
            pearsons_map[ions_type].append(0)
            pearsons_map["all"].append(0)
            continue

        if (np.max(light_ions_xic["intensity"]) > 0 and
                np.max(heavy_ions_xic["intensity"]) > 0):
            intensitys_map[ions_type] += np.sum(light_ions_xic["intensity"])
            intensitys_map[ions_type] += np.sum(heavy_ions_xic["intensity"])
            intensitys_map["all"] = light_all_intensity + \
                heavy_all_intensity

        pearson_corr = calc_xic_score(light_ions_xic, heavy_ions_xic)

        pearsons_map[ions_type].append(pearson_corr)
        pearsons_map["all"].append(pearson_corr)

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

    # 分别提取出b离子，y离子，全部的三种特征
    for key, value_list in pearsons_map.items():
        # logging.info(f"{key} {value_list}")
        stats = extract_ion_pearson_features(value_list)
        # logging.info(stats)

        features[f"{key}_count"] = stats["count"]
        features[f"{key}_p25"] = stats["p25"]
        features[f"{key}_p50"] = stats["p50"]  # 关键特征！
        features[f"{key}_p75"] = stats["p75"]
        features[f"{key}_mean"] = stats["mean"]

    features["matched_intensity_percent"] = (
        (intensitys_map["b"] + intensitys_map["y"]) / intensitys_map["all"])
    return features


def extract_ion_pearson_features(ions_pearsons: []) -> dict:
    """
    计算出这个数组中的25%,50%,75% 分位数，和均值
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
        }

    p25 = np.clip(np.percentile(clean_vals, 25), 0, 1)
    p50 = np.clip(np.percentile(clean_vals, 50), 0, 1)
    p75 = np.clip(np.percentile(clean_vals, 75), 0, 1)
    mean = np.mean(clean_vals)

    return {
        "count": count,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "mean": mean,
    }


def calc_xic_score(
    light_xic: np.array, heavy_xic: np.array
) -> np.float32:
    """ 根据mono 的XIC 计算出相似度打分 """

    # 计算重标平均误差
    mz_avg_err = np.average(heavy_xic["ppm_error"])

    # 计算峰顶的时间差
    rt_apex_light = light_xic["rt"][np.argmax(light_xic["intensity"])]
    rt_apex_heavy = heavy_xic["rt"][np.argmax(heavy_xic["intensity"])]
    apex_delta = abs(rt_apex_light - rt_apex_heavy)

    # 计算峰相关形
    # 统一时间轴
    common_rt = np.linspace(
        max(light_xic["rt"].min(), heavy_xic["rt"].min()),
        min(light_xic["rt"].max(), heavy_xic["rt"].max()),
        100)
    inten1_interp = interp(common_rt, light_xic["rt"], light_xic["intensity"])
    inten2_interp = interp(common_rt, heavy_xic["rt"], heavy_xic["intensity"])

    corr, _ = pearsonr(inten1_interp, inten2_interp)

    # logging.info(f"mz_avg_err : {mz_avg_err}, apex_delta:{
    #              apex_delta}, corr:{corr}")

    if np.isnan(corr):
        return 0

    return corr


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
