
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

    if psm._sequence != "ALSSQHQAR":
        return
    # TODO:

    logging.info(f"处理信息 {psm}")

    # 从配置中获得 ppm
    mass_tol_ppm = config[ConfigKeys.GENERAL].getfloat(ConfigKeys.MASS_TOL_PPM)

    light_xic = dia_data.xic_peaks_extreact(
        psm._rt_start - 0.3, psm._rt_stop + 0.3,
        psm._precursor_mz, mass_tol_ppm)

    heavy_precursor_mz, fragment_ions = psm.get_heavy_info(HeavyType.SILAC)

    logging.info(f"light_mz: {psm._precursor_mz},heavy_mz: {
                 heavy_precursor_mz}")

    heavy_xic = dia_data.xic_peaks_extreact(
        psm._rt_start - 0.3, psm._rt_stop + 0.3,
        heavy_precursor_mz, mass_tol_ppm)

    # 进行画图
    plot_light_heavy_xic(light_xic, heavy_xic)

    # 计算出 person_corr
    person_corr = calc_xic_score(light_xic, heavy_xic)

    res_corr = []  # 用 list 收集

    # 枚举所有的信息
    for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:

        # 说明这个碎片离子不受到重标的影响
        if np.abs(heavy_mass - light_mass) < 0.01:
            continue

        # 计算出 light 信息
        light_ions_xic = dia_data.xic_ms2_peaks_extract(
            psm._rt_start - 0.3, psm._rt_stop + 0.3,
            precursor_mz=psm._precursor_mz,
            ions_mass=light_mass,
            mass_tol_ppm=mass_tol_ppm
        )

        # 计算出 heavy 信息
        heavy_ions_xic = dia_data.xic_ms2_peaks_extract(
            psm._rt_start - 0.3, psm._rt_stop + 0.3,
            precursor_mz=heavy_precursor_mz,
            ions_mass=heavy_mass,
            mass_tol_ppm=mass_tol_ppm
        )

        pearson_corr = calc_xic_score(light_ions_xic, heavy_ions_xic)

        res_corr.append(pearson_corr)  # 循环里追加

        # logging.info(f"{ions_type} {ions_num} : person({person_corr})")

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

    res_corr = np.array(res_corr)  # 转成 numpy 数组
    ms2_count = np.sum(res_corr > 0.8)

    if ms2_count >= 3:
        logging.info(f"true: {psm}")
    else:
        logging.info(f"false: {psm}")
    exit(0)


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
