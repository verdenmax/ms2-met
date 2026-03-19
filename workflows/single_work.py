
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from numpy import interp

from spectrum.psm_info import PSMInfo
from spectrum.psm_info import HeavyType
from spectrum.psm_info import get_SILAC_increase_mass
from spectrum.dia_data import DIAData

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
        features["precursor_mz_avg_err"] = 0.0
        features["precursor_light_max_int"] = 0.0
        features["precursor_heavy_max_int"] = 0.0
        features["precursor_intensity_ratio"] = 0.0
    else:
        precursor_score = calc_xic_score(light_xic, heavy_xic)
        features["precursor_pearson"] = precursor_score["pearson"]
        features["precursor_apex_delta"] = precursor_score["apex_delta"]
        features["precursor_mz_avg_err"] = precursor_score["mz_avg_err"]
        features["precursor_light_max_int"] = precursor_score["light_max_int"]
        features["precursor_heavy_max_int"] = precursor_score["heavy_max_int"]
        features["precursor_intensity_ratio"] = precursor_score["intensity_ratio"]

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

    fragment_apex_deltas = []
    fragment_mz_errs = []

    _, fragment_ions = psm1.get_heavy_info(HeavyType.SILAC)
    # 枚举所有的信息
    for ions_type, ions_num, light_mass, _ in fragment_ions:

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

        ion_score = calc_xic_score(light_ions_xic, heavy_ions_xic)

        pearsons_map[ions_type].append(ion_score["pearson"])
        pearsons_map["all"].append(ion_score["pearson"])
        fragment_apex_deltas.append(ion_score["apex_delta"])
        fragment_mz_errs.append(ion_score["mz_avg_err"])

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

    features["matched_intensity_percent"] = (
        (intensitys_map["b"] + intensitys_map["y"]) / intensitys_map["all"])

    # 碎片级 apex_delta / mz_err 汇总
    features.update(extract_ion_numeric_features(
        fragment_apex_deltas, "all_apex_delta"))
    features.update(extract_ion_numeric_features(
        fragment_mz_errs, "all_mz_err"))

    # 序列级特征
    features["kr_count"] = psm1._sequence.count('K') + \
        psm1._sequence.count('R')
    features["modification_count"] = len(psm1._modify)
    features["total_silac_shift"] = get_SILAC_increase_mass(psm1._sequence)

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
        features["precursor_apex_delta"] = 0.0
        features["precursor_mz_avg_err"] = 0.0
        features["precursor_light_max_int"] = 0.0
        features["precursor_heavy_max_int"] = 0.0
        features["precursor_intensity_ratio"] = 0.0
    else:
        precursor_score = calc_xic_score(light_xic, heavy_xic)
        features["precursor_pearson"] = precursor_score["pearson"]
        features["precursor_apex_delta"] = precursor_score["apex_delta"]
        features["precursor_mz_avg_err"] = precursor_score["mz_avg_err"]
        features["precursor_light_max_int"] = precursor_score["light_max_int"]
        features["precursor_heavy_max_int"] = precursor_score["heavy_max_int"]
        features["precursor_intensity_ratio"] = precursor_score["intensity_ratio"]

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

    heavy_in_raw = dia_data.check_in_raw(heavy_precursor_mz)

    fragment_apex_deltas = []
    fragment_mz_errs = []

    ion_data = []  # 存储每个离子的完整数据
    # 枚举所有的信息
    for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:

        if not heavy_in_raw:
            continue

        # 如果在相同的区间，并且质量相同，说明重标不影响该碎片离子
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

        ion_score = calc_xic_score(light_ions_xic, heavy_ions_xic)

        pearsons_map[ions_type].append(ion_score["pearson"])
        pearsons_map["all"].append(ion_score["pearson"])
        fragment_apex_deltas.append(ion_score["apex_delta"])
        fragment_mz_errs.append(ion_score["mz_avg_err"])

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

    features["matched_intensity_percent"] = (
        (intensitys_map["b"] + intensitys_map["y"]) / intensitys_map["all"])

    # 碎片级 apex_delta / mz_err 汇总
    features.update(extract_ion_numeric_features(
        fragment_apex_deltas, "all_apex_delta"))
    features.update(extract_ion_numeric_features(
        fragment_mz_errs, "all_mz_err"))

    # 序列级特征
    features["kr_count"] = psm._sequence.count('K') + \
        psm._sequence.count('R')
    features["modification_count"] = len(psm._modify)
    features["total_silac_shift"] = get_SILAC_increase_mass(psm._sequence)

    if heavy_in_raw:
        features["heavy_in_raw"] = 1
    else:
        features["heavy_in_raw"] = 0

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


def extract_ion_numeric_features(values: list, prefix: str) -> dict:
    """
    对碎片级数值列表（如 apex_delta、mz_err）计算均值和中位数。
    清除 NaN/Inf 值后统计。
    """
    clean_vals = [v for v in values if not np.isnan(v) and np.isfinite(v)]
    if len(clean_vals) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(clean_vals)),
        f"{prefix}_p50": float(np.median(clean_vals)),
    }


def _default_xic_score() -> dict:
    """calc_xic_score 的全零默认返回值"""
    return {
        "pearson": np.float32(0.0),
        "mz_avg_err": 0.0,
        "apex_delta": 0.0,
        "light_max_int": 0.0,
        "heavy_max_int": 0.0,
        "intensity_ratio": 0.0,
    }


def calc_xic_score(
    light_xic: np.array, heavy_xic: np.array,
    intensity_threshold: float = 1e-10
) -> dict:
    """ 根据轻重标 XIC 计算综合特征，返回包含 pearson/mz_avg_err/apex_delta/强度信息的字典 """

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

    # 计算强度信息
    light_max_int = float(np.max(light_xic["intensity"]))
    heavy_max_int = float(np.max(heavy_xic["intensity"]))
    light_total = float(np.sum(light_xic["intensity"]))
    heavy_total = float(np.sum(heavy_xic["intensity"]))
    intensity_ratio = light_total / heavy_total if heavy_total > 0 else 0.0

    # 计算峰相关性
    # 统一时间轴
    rt_start = max(light_xic["rt"].min(), heavy_xic["rt"].min())
    rt_end = min(light_xic["rt"].max(), heavy_xic["rt"].max())

    if rt_start >= rt_end:
        result = _default_xic_score()
        result["mz_avg_err"] = mz_avg_err
        result["apex_delta"] = apex_delta
        result["light_max_int"] = light_max_int
        result["heavy_max_int"] = heavy_max_int
        result["intensity_ratio"] = intensity_ratio
        return result

    common_rt = np.linspace(rt_start, rt_end, 100)
    inten1_interp = interp(common_rt, light_xic["rt"], light_xic["intensity"])
    inten2_interp = interp(common_rt, heavy_xic["rt"], heavy_xic["intensity"])

    # 检查是否都是0或接近0
    light_near_zero = np.all(np.abs(inten1_interp) < intensity_threshold)
    heavy_near_zero = np.all(np.abs(inten2_interp) < intensity_threshold)

    if light_near_zero and heavy_near_zero:
        corr = 0.0
    elif light_near_zero or heavy_near_zero:
        corr = 0.0
    else:
        if np.std(inten1_interp) < 1e-10 or np.std(inten2_interp) < 1e-10:
            corr = 0.0
        else:
            try:
                corr, _ = pearsonr(inten1_interp, inten2_interp)
            except (ValueError, RuntimeWarning):
                corr = 0.0

    return {
        "pearson": np.float32(corr),
        "mz_avg_err": mz_avg_err,
        "apex_delta": apex_delta,
        "light_max_int": light_max_int,
        "heavy_max_int": heavy_max_int,
        "intensity_ratio": intensity_ratio,
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
