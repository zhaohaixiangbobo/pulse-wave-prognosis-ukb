"""
文件说明：为 p4205_i0 单周期脉冲波提取高级统计、时域、形态、频域和非线性特征。

p4205_i0 只有固定 100 个采样点且缺少真实采样率，因此时间使用周期比例，频率使用
cycles/sample 归一化频率。条件特征检测失败时返回 NaN，并通过质量标记说明失败原因。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample, savgol_filter
from scipy.stats import kurtosis, skew


EPSILON = 1e-12
RESAMPLED_POINTS = 200
QUALITY_COLUMNS = [
    "p4205_q_parse_success",
    "p4205_q_expected_length",
    "p4205_q_nonflat",
    "p4205_q_h1_success",
    "p4205_q_h3_success",
    "p4205_q_h4_success",
    "p4205_q_h5_success",
    "p4205_q_width_success",
]


@dataclass(frozen=True)
class FeatureSpec:
    """高级波形特征的目录定义。"""

    name: str
    category: str
    meaning: str
    unit: str
    valid_when: str = "波形解析成功且非平坦"


def _specs() -> list[FeatureSpec]:
    specs = [
        FeatureSpec(name, "quality", meaning, "0/1", "总是输出")
        for name, meaning in [
            ("p4205_q_parse_success", "波形解析成功"),
            ("p4205_q_expected_length", "解析长度等于声明长度且为100"),
            ("p4205_q_nonflat", "波形具有有效幅度范围"),
            ("p4205_q_h1_success", "主波峰检测成功"),
            ("p4205_q_h3_success", "重搏前波检测成功"),
            ("p4205_q_h4_success", "降中峡检测成功"),
            ("p4205_q_h5_success", "重搏波检测成功"),
            ("p4205_q_width_success", "主波上1/3宽度检测成功"),
        ]
    ]
    specs += [
        FeatureSpec(name, "statistical", meaning, unit)
        for name, meaning, unit in [
            ("p4205_stat_raw_mean", "原始幅值均值", "原始任意单位"),
            ("p4205_stat_raw_std", "原始幅值标准差", "原始任意单位"),
            ("p4205_stat_raw_median", "原始幅值中位数", "原始任意单位"),
            ("p4205_stat_raw_mad", "原始幅值中位绝对偏差", "原始任意单位"),
            ("p4205_stat_raw_p05", "原始幅值第5百分位", "原始任意单位"),
            ("p4205_stat_raw_p25", "原始幅值第25百分位", "原始任意单位"),
            ("p4205_stat_raw_p75", "原始幅值第75百分位", "原始任意单位"),
            ("p4205_stat_raw_p95", "原始幅值第95百分位", "原始任意单位"),
            ("p4205_stat_raw_iqr", "原始幅值四分位距", "原始任意单位"),
            ("p4205_stat_raw_range", "原始幅值极差", "原始任意单位"),
            ("p4205_stat_raw_cv", "原始幅值变异系数", "比值"),
            ("p4205_stat_norm_skew", "鲁棒归一化波形偏度", "无量纲"),
            ("p4205_stat_norm_kurtosis", "鲁棒归一化波形峰度", "无量纲"),
            ("p4205_stat_norm_rms", "鲁棒归一化波形均方根", "无量纲"),
            ("p4205_stat_norm_energy", "鲁棒归一化波形能量", "无量纲"),
        ]
    ]
    time_items = [
        ("foot_pos_ratio", "起点位置比例", "周期比例", "h1检测成功"),
        ("end_pos_ratio", "终点位置比例", "周期比例", "h1检测成功"),
        ("h1", "主波峰幅值", "鲁棒归一化幅值", "h1检测成功"),
        ("h3", "重搏前波幅值", "鲁棒归一化幅值", "h3检测成功"),
        ("h4", "降中峡幅值", "鲁棒归一化幅值", "h4检测成功"),
        ("h5", "重搏波幅值", "鲁棒归一化幅值", "h5检测成功"),
        ("t1_ratio", "起点至主峰时间比例", "周期比例", "h1检测成功"),
        ("t3_ratio", "起点至重搏前波时间比例", "周期比例", "h3检测成功"),
        ("t4_ratio", "起点至降中峡时间比例", "周期比例", "h4检测成功"),
        ("t5_ratio", "降中峡至终点时间比例", "周期比例", "h4检测成功"),
        ("width_one_third_ratio", "主波上1/3高度宽度", "周期比例", "宽度检测成功"),
        ("systolic_area", "收缩期面积", "归一化面积", "h4检测成功"),
        ("diastolic_area", "舒张期面积", "归一化面积", "h4检测成功"),
        ("h3_h1_ratio", "重搏前波与主峰幅值比", "比值", "h3检测成功"),
        ("h4_h1_ratio", "降中峡与主峰幅值比", "比值", "h4检测成功"),
        ("h5_h1_ratio", "重搏波与主峰幅值比", "比值", "h5检测成功"),
        ("width_period_ratio", "主波宽度与周期比", "比值", "宽度检测成功"),
        ("systolic_diastolic_area_ratio", "收缩与舒张面积比", "比值", "h4检测成功"),
    ]
    specs += [
        FeatureSpec(f"p4205_time_{name}", "time_domain", meaning, unit, valid)
        for name, meaning, unit, valid in time_items
    ]
    specs += [
        FeatureSpec(f"p4205_morph_{name}", "morphology", meaning, unit)
        for name, meaning, unit in [
            ("max_rise", "最大一阶上升变化", "归一化幅值/点"),
            ("max_fall", "最大一阶下降变化", "归一化幅值/点"),
            ("max_second_derivative", "最大二阶变化", "归一化幅值/点平方"),
            ("min_second_derivative", "最小二阶变化", "归一化幅值/点平方"),
            ("curvature_mean_abs", "平均绝对曲率", "无量纲"),
            ("inflection_count", "二阶导数过零数量", "计数"),
            ("peak_count", "局部峰数量", "计数"),
            ("valley_count", "局部谷数量", "计数"),
            ("shoulder_strength", "下降支肩部相对强度", "无量纲"),
            ("systolic_slope", "起点至主峰平均斜率", "归一化幅值/周期"),
            ("diastolic_slope", "主峰至终点平均斜率", "归一化幅值/周期"),
            ("symmetry_ratio", "上升段与下降段时长比", "比值"),
        ]
    ]
    freq_items = [
        ("dominant_freq_norm", "主频", "cycles/sample"),
        ("total_power", "总频谱功率", "无量纲"),
        ("low_power", "低频带功率[0,0.1)", "无量纲"),
        ("mid_power", "中频带功率[0.1,0.25)", "无量纲"),
        ("high_power", "高频带功率[0.25,0.5]", "无量纲"),
        ("low_total_ratio", "低频功率占比", "比值"),
        ("mid_total_ratio", "中频功率占比", "比值"),
        ("high_total_ratio", "高频功率占比", "比值"),
        ("low_high_ratio", "低频与高频功率比", "比值"),
        ("spectral_centroid_norm", "谱质心", "cycles/sample"),
        ("spectral_entropy", "归一化谱熵", "0-1"),
    ]
    specs += [
        FeatureSpec(f"p4205_freq_{name}", "frequency", meaning, unit)
        for name, meaning, unit in freq_items
    ]
    for rank in range(1, 6):
        specs += [
            FeatureSpec(f"p4205_freq_peak{rank}_freq_norm", "frequency", f"第{rank}频谱峰频率", "cycles/sample"),
            FeatureSpec(f"p4205_freq_peak{rank}_amplitude", "frequency", f"第{rank}频谱峰幅值", "无量纲"),
            FeatureSpec(f"p4205_freq_peak{rank}_phase_rad", "frequency", f"第{rank}频谱峰相位", "弧度"),
            FeatureSpec(f"p4205_freq_peak{rank}_rel_amplitude", "frequency", f"第{rank}频谱峰相对第一峰幅值", "比值"),
        ]
    specs += [
        FeatureSpec(f"p4205_nonlinear_{name}", "nonlinear", meaning, unit)
        for name, meaning, unit in [
            ("sample_entropy", "样本熵，单周期短序列探索性指标", "无量纲"),
            ("recurrence_rate", "递归率，单周期短序列探索性指标", "比值"),
            ("determinism", "确定率，单周期短序列探索性指标", "比值"),
            ("mean_diag_length", "递归图平均对角线长度", "点"),
            ("max_diag_length", "递归图最大对角线长度", "点"),
            ("largest_lyapunov", "简化最大Lyapunov指数", "每点"),
        ]
    ]
    return specs


ADVANCED_FEATURE_SPECS = _specs()
ADVANCED_FEATURE_COLUMNS = [spec.name for spec in ADVANCED_FEATURE_SPECS]


def feature_catalog_frame() -> pd.DataFrame:
    """返回可直接导出的高级特征目录。"""
    return pd.DataFrame([spec.__dict__ for spec in ADVANCED_FEATURE_SPECS])


def parse_waveform_xy(value: Any) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    """解析波形并同时保留横坐标、幅值和声明点数。"""
    if not isinstance(value, str) or "|" not in value:
        return None, None, None
    parts = value.strip().split("|")
    try:
        declared = int(float(parts[0]))
    except (TypeError, ValueError):
        declared = None
    x_values: list[float] = []
    y_values: list[float] = []
    for point in parts[1:]:
        if "," not in point:
            continue
        raw_x, raw_y = point.split(",", 1)
        try:
            x_value, y_value = float(raw_x), float(raw_y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            x_values.append(x_value)
            y_values.append(y_value)
    if not y_values:
        return None, None, declared
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), declared


def robust_normalize(signal: np.ndarray) -> tuple[np.ndarray | None, float]:
    """使用中位数与 P95-P5 做鲁棒归一化，平坦波形返回 None。"""
    values = np.asarray(signal, dtype=float)
    scale = float(np.percentile(values, 95) - np.percentile(values, 5))
    if not math.isfinite(scale) or scale <= EPSILON:
        return None, scale
    return (values - float(np.median(values))) / scale, scale


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if math.isfinite(denominator) and abs(denominator) > EPSILON else np.nan


def _blank_features() -> dict[str, float]:
    row = {name: np.nan for name in ADVANCED_FEATURE_COLUMNS}
    row.update({name: 0.0 for name in QUALITY_COLUMNS})
    return row


def _detect_landmarks(signal: np.ndarray) -> dict[str, int | None]:
    """在 200 点平滑波形上检测主峰、降中峡、重搏前波和重搏波。"""
    smooth = savgol_filter(signal, window_length=11, polyorder=3)
    n = smooth.size
    foot = int(np.argmin(smooth[: n // 4]))
    h1_search_end = max(foot + 2, int(0.60 * n))
    h1 = foot + int(np.argmax(smooth[foot:h1_search_end]))
    end_start = int(0.75 * n)
    end = end_start + int(np.argmin(smooth[end_start:]))
    if not (foot < h1 < end) or smooth[h1] - smooth[foot] < 0.15:
        return {"foot": foot, "h1": None, "h3": None, "h4": None, "h5": None, "end": end}

    h4: int | None = None
    second = savgol_filter(smooth, window_length=11, polyorder=3, deriv=2)
    region_start, region_end = h1 + 4, min(end, int(0.88 * n))
    if region_end > region_start + 4:
        region = second[region_start:region_end]
        peaks, properties = find_peaks(region, prominence=max(0.05 * float(np.std(second)), EPSILON))
        if peaks.size:
            h4 = region_start + int(peaks[np.argmax(properties["prominences"])])

    h3: int | None = None
    h5: int | None = None
    if h4 is not None and h4 > h1 + 6:
        region = smooth[h1 + 3 : h4]
        prominence = max(0.04 * float(smooth[h1] - smooth[h4]), EPSILON)
        peaks, _ = find_peaks(region, prominence=prominence, distance=4)
        if peaks.size:
            h3 = h1 + 3 + int(peaks[np.argmax(region[peaks])])

        h5_end = min(end, h4 + int(0.15 * n))
        region = smooth[h4:h5_end]
        peaks, _ = find_peaks(region, prominence=max(0.02 * float(smooth[h1] - smooth[h4]), EPSILON), distance=4)
        if peaks.size:
            h5 = h4 + int(peaks[np.argmax(region[peaks])])
    return {"foot": foot, "h1": h1, "h3": h3, "h4": h4, "h5": h5, "end": end}


def _width_at_one_third(signal: np.ndarray, foot: int, h1: int, end: int) -> float:
    level = float(signal[foot] + (signal[h1] - signal[foot]) / 3.0)
    left_candidates = np.where(signal[foot : h1 + 1] >= level)[0]
    right_candidates = np.where(signal[h1 : end + 1] <= level)[0]
    if not left_candidates.size or not right_candidates.size:
        return np.nan
    left = foot + int(left_candidates[0])
    right = h1 + int(right_candidates[0])
    return float(right - left) if right > left else np.nan


def _sample_entropy(signal: np.ndarray, m: int = 2) -> float:
    std = float(np.std(signal))
    if signal.size < m + 3 or std <= EPSILON:
        return np.nan
    tolerance = 0.2 * std

    def count_matches(length: int) -> int:
        embedded = np.lib.stride_tricks.sliding_window_view(signal, length)
        distances = np.max(np.abs(embedded[:, None, :] - embedded[None, :, :]), axis=2)
        return int(np.sum(np.triu(distances <= tolerance, k=1)))

    count_m = count_matches(m)
    count_m1 = count_matches(m + 1)
    if count_m == 0 or count_m1 == 0:
        return np.nan
    return float(-np.log(count_m1 / count_m))


def _diagonal_lengths(matrix: np.ndarray) -> list[int]:
    lengths: list[int] = []
    for offset in range(-matrix.shape[0] + 1, matrix.shape[1]):
        diagonal = np.diagonal(matrix, offset=offset)
        padded = np.pad(diagonal.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        runs = np.where(changes == -1)[0] - np.where(changes == 1)[0]
        lengths.extend(int(value) for value in runs if value >= 2)
    return lengths


def _rqa_features(signal: np.ndarray) -> dict[str, float]:
    if signal.size < 20 or float(np.std(signal)) <= EPSILON:
        return {}
    embedded = np.lib.stride_tricks.sliding_window_view(signal, 3)
    distances = np.max(np.abs(embedded[:, None, :] - embedded[None, :, :]), axis=2)
    recurrence = distances <= 0.15 * float(np.std(signal))
    np.fill_diagonal(recurrence, False)
    recurrence_points = int(recurrence.sum())
    if recurrence_points == 0:
        return {}
    lengths = _diagonal_lengths(recurrence)
    diagonal_points = int(sum(lengths))
    return {
        "p4205_nonlinear_recurrence_rate": float(recurrence.mean()),
        "p4205_nonlinear_determinism": float(diagonal_points / recurrence_points),
        "p4205_nonlinear_mean_diag_length": float(np.mean(lengths)) if lengths else np.nan,
        "p4205_nonlinear_max_diag_length": float(max(lengths)) if lengths else np.nan,
    }


def _largest_lyapunov(signal: np.ndarray, max_steps: int = 8) -> float:
    """使用确定性的最近邻发散斜率估计短序列 Lyapunov 指数。"""
    if signal.size < 30 or float(np.std(signal)) <= EPSILON:
        return np.nan
    embedded = np.lib.stride_tricks.sliding_window_view(signal, 3)
    distances = np.linalg.norm(embedded[:, None, :] - embedded[None, :, :], axis=2)
    temporal = np.abs(np.arange(len(embedded))[:, None] - np.arange(len(embedded))[None, :])
    distances[temporal <= 3] = np.inf
    neighbors = np.argmin(distances, axis=1)
    curves: list[np.ndarray] = []
    for index, neighbor in enumerate(neighbors):
        steps = min(max_steps, len(embedded) - index, len(embedded) - neighbor)
        if steps < 4 or not np.isfinite(distances[index, neighbor]):
            continue
        divergence = np.linalg.norm(embedded[index : index + steps] - embedded[neighbor : neighbor + steps], axis=1)
        if np.all(divergence > EPSILON):
            curves.append(np.log(divergence))
    if not curves:
        return np.nan
    slopes = [np.polyfit(np.arange(len(curve)), curve, 1)[0] for curve in curves]
    positive = [float(value) for value in slopes if math.isfinite(value) and value > 0]
    return float(np.median(positive)) if positive else np.nan


def extract_advanced_features(value: Any, include_nonlinear: bool = True) -> dict[str, float]:
    """从一个 p4205_i0 字符串提取全部高级特征。"""
    row = _blank_features()
    x_values, raw, declared = parse_waveform_xy(value)
    if raw is None or x_values is None:
        return row
    row["p4205_q_parse_success"] = 1.0
    row["p4205_q_expected_length"] = float(declared == 100 and raw.size == 100 and x_values.size == 100)
    normalized, _ = robust_normalize(raw)
    if normalized is None:
        return row
    row["p4205_q_nonflat"] = 1.0

    p05, p25, p75, p95 = np.percentile(raw, [5, 25, 75, 95])
    raw_mean, raw_std, raw_median = float(np.mean(raw)), float(np.std(raw)), float(np.median(raw))
    row.update(
        {
            "p4205_stat_raw_mean": raw_mean,
            "p4205_stat_raw_std": raw_std,
            "p4205_stat_raw_median": raw_median,
            "p4205_stat_raw_mad": float(np.median(np.abs(raw - raw_median))),
            "p4205_stat_raw_p05": float(p05),
            "p4205_stat_raw_p25": float(p25),
            "p4205_stat_raw_p75": float(p75),
            "p4205_stat_raw_p95": float(p95),
            "p4205_stat_raw_iqr": float(p75 - p25),
            "p4205_stat_raw_range": float(np.ptp(raw)),
            "p4205_stat_raw_cv": _safe_ratio(raw_std, abs(raw_mean)),
            "p4205_stat_norm_skew": float(skew(normalized, bias=False)),
            "p4205_stat_norm_kurtosis": float(kurtosis(normalized, bias=False)),
            "p4205_stat_norm_rms": float(np.sqrt(np.mean(normalized**2))),
            "p4205_stat_norm_energy": float(np.sum(normalized**2)),
        }
    )

    detected = savgol_filter(resample(normalized, RESAMPLED_POINTS), 11, 3)
    landmarks = _detect_landmarks(detected)
    foot, h1, h3, h4, h5, end = (landmarks[name] for name in ("foot", "h1", "h3", "h4", "h5", "end"))
    for key, point in (("h1", h1), ("h3", h3), ("h4", h4), ("h5", h5)):
        row[f"p4205_q_{key}_success"] = float(point is not None)
    if h1 is not None and end is not None:
        period = float(end - foot)
        row.update(
            {
                "p4205_time_foot_pos_ratio": foot / (RESAMPLED_POINTS - 1),
                "p4205_time_end_pos_ratio": end / (RESAMPLED_POINTS - 1),
                "p4205_time_h1": float(detected[h1]),
                "p4205_time_t1_ratio": _safe_ratio(h1 - foot, period),
            }
        )
        width = _width_at_one_third(detected, foot, h1, end)
        row["p4205_q_width_success"] = float(math.isfinite(width))
        row["p4205_time_width_one_third_ratio"] = _safe_ratio(width, period)
        row["p4205_time_width_period_ratio"] = _safe_ratio(width, period)
        row["p4205_morph_systolic_slope"] = _safe_ratio(float(detected[h1] - detected[foot]), h1 - foot)
        row["p4205_morph_diastolic_slope"] = _safe_ratio(float(detected[end] - detected[h1]), end - h1)
        row["p4205_morph_symmetry_ratio"] = _safe_ratio(h1 - foot, end - h1)
        if h3 is not None:
            row["p4205_time_h3"] = float(detected[h3])
            row["p4205_time_t3_ratio"] = _safe_ratio(h3 - foot, period)
            row["p4205_time_h3_h1_ratio"] = _safe_ratio(float(detected[h3]), float(detected[h1]))
        if h4 is not None:
            row["p4205_time_h4"] = float(detected[h4])
            row["p4205_time_t4_ratio"] = _safe_ratio(h4 - foot, period)
            row["p4205_time_t5_ratio"] = _safe_ratio(end - h4, period)
            row["p4205_time_h4_h1_ratio"] = _safe_ratio(float(detected[h4]), float(detected[h1]))
            systolic = float(np.trapezoid(detected[foot : h4 + 1] - detected[foot]))
            diastolic = float(np.trapezoid(detected[h4 : end + 1] - detected[end]))
            row["p4205_time_systolic_area"] = systolic
            row["p4205_time_diastolic_area"] = diastolic
            row["p4205_time_systolic_diastolic_area_ratio"] = _safe_ratio(systolic, diastolic)
            row["p4205_morph_shoulder_strength"] = _safe_ratio(
                float(detected[h4] - detected[end]), float(detected[h1] - detected[end])
            )
        if h5 is not None:
            row["p4205_time_h5"] = float(detected[h5])
            row["p4205_time_h5_h1_ratio"] = _safe_ratio(float(detected[h5]), float(detected[h1]))

    diff1 = np.diff(normalized)
    diff2 = np.diff(normalized, n=2)
    peaks, _ = find_peaks(normalized, prominence=max(0.02 * float(np.ptp(normalized)), EPSILON), distance=3)
    valleys, _ = find_peaks(-normalized, prominence=max(0.02 * float(np.ptp(normalized)), EPSILON), distance=3)
    row.update(
        {
            "p4205_morph_max_rise": float(np.max(diff1)),
            "p4205_morph_max_fall": float(np.min(diff1)),
            "p4205_morph_max_second_derivative": float(np.max(diff2)),
            "p4205_morph_min_second_derivative": float(np.min(diff2)),
            "p4205_morph_curvature_mean_abs": float(np.mean(np.abs(diff2) / (1 + diff1[:-1] ** 2) ** 1.5)),
            "p4205_morph_inflection_count": float(np.sum(np.sign(diff2[1:]) != np.sign(diff2[:-1]))),
            "p4205_morph_peak_count": float(peaks.size),
            "p4205_morph_valley_count": float(valleys.size),
        }
    )

    centered = normalized - float(np.mean(normalized))
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(centered.size, d=1.0)
    amplitudes = np.abs(spectrum)
    power = amplitudes**2
    power[0] = 0.0
    total_power = float(np.sum(power))
    positive_indices = np.arange(1, len(frequencies))
    dominant_index = int(positive_indices[np.argmax(power[positive_indices])])
    low = frequencies < 0.10
    mid = (frequencies >= 0.10) & (frequencies < 0.25)
    high = frequencies >= 0.25
    low_power, mid_power, high_power = (float(np.sum(power[mask])) for mask in (low, mid, high))
    probabilities = power / total_power if total_power > EPSILON else np.zeros_like(power)
    nonzero = probabilities > 0
    entropy = -float(np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))
    entropy /= math.log2(max(2, int(nonzero.sum())))
    row.update(
        {
            "p4205_freq_dominant_freq_norm": float(frequencies[dominant_index]),
            "p4205_freq_total_power": total_power,
            "p4205_freq_low_power": low_power,
            "p4205_freq_mid_power": mid_power,
            "p4205_freq_high_power": high_power,
            "p4205_freq_low_total_ratio": _safe_ratio(low_power, total_power),
            "p4205_freq_mid_total_ratio": _safe_ratio(mid_power, total_power),
            "p4205_freq_high_total_ratio": _safe_ratio(high_power, total_power),
            "p4205_freq_low_high_ratio": _safe_ratio(low_power, high_power),
            "p4205_freq_spectral_centroid_norm": _safe_ratio(float(np.sum(frequencies * power)), total_power),
            "p4205_freq_spectral_entropy": entropy,
        }
    )
    spectral_peaks, _ = find_peaks(amplitudes[1:])
    candidates = spectral_peaks + 1 if spectral_peaks.size else positive_indices
    ranked = candidates[np.argsort(amplitudes[candidates])[::-1]][:5]
    first_amplitude = float(amplitudes[ranked[0]]) if ranked.size else np.nan
    for rank in range(1, 6):
        if rank <= ranked.size:
            index = int(ranked[rank - 1])
            row[f"p4205_freq_peak{rank}_freq_norm"] = float(frequencies[index])
            row[f"p4205_freq_peak{rank}_amplitude"] = float(amplitudes[index])
            row[f"p4205_freq_peak{rank}_phase_rad"] = float(np.angle(spectrum[index]))
            row[f"p4205_freq_peak{rank}_rel_amplitude"] = _safe_ratio(float(amplitudes[index]), first_amplitude)

    if include_nonlinear:
        row["p4205_nonlinear_sample_entropy"] = _sample_entropy(normalized)
        row.update(_rqa_features(normalized))
        row["p4205_nonlinear_largest_lyapunov"] = _largest_lyapunov(normalized)
    return row


def build_advanced_feature_frame(series: pd.Series, include_nonlinear: bool = True) -> pd.DataFrame:
    """批量提取高级特征，并保持与输入行顺序一致。"""
    rows = [extract_advanced_features(value, include_nonlinear=include_nonlinear) for value in series.tolist()]
    return pd.DataFrame(rows, index=series.index).reindex(columns=ADVANCED_FEATURE_COLUMNS)


def quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """汇总检测成功率、特征缺失率和无限值数量。"""
    numeric = frame.replace([np.inf, -np.inf], np.nan)
    return {
        "row_count": int(len(frame)),
        "quality_success_rates": {
            column: float(pd.to_numeric(frame[column], errors="coerce").mean()) for column in QUALITY_COLUMNS
        },
        "feature_missing_rates": {column: float(numeric[column].isna().mean()) for column in frame.columns},
        "infinite_value_count": int(np.isinf(frame.select_dtypes(include=[np.number]).to_numpy()).sum()),
    }
