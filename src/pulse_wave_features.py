"""核心实现归档副本；来源见SOURCE_MANIFEST.json。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


WAVE_FEATURE_COLUMNS = [
    "p4205_signal_length",
    "p4205_declared_points",
    "p4205_parse_ratio",
    "p4205_mean",
    "p4205_std",
    "p4205_max",
    "p4205_min",
    "p4205_median",
    "p4205_skew",
    "p4205_kurtosis",
    "p4205_range",
    "p4205_iqr",
    "p4205_peak_count",
    "p4205_main_peak_pos",
    "p4205_main_peak_value",
    "p4205_max_rise",
    "p4205_max_fall",
    "p4205_area",
    "p4205_energy",
    "p4205_zero_crossing_count",
    "p4205_fft_mean",
    "p4205_fft_std",
    "p4205_fft_max",
    "p4205_dominant_freq",
    "p4205_spectral_energy",
    "p4205_spectral_entropy",
]


def default_wave_features() -> dict[str, float]:
    """返回波形特征默认值，保证异常样本不打断批量处理。"""
    return {column: np.nan for column in WAVE_FEATURE_COLUMNS}


def safe_float(value: Any) -> float | None:
    """将单个值安全转换为 float，失败时返回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_waveform(wave_str: Any) -> tuple[np.ndarray | None, int | None]:
    """解析 `100|0,0|1,88|...` 格式的波形字符串。"""
    if not isinstance(wave_str, str):
        return None, None

    text = wave_str.strip()
    if not text or "|" not in text:
        return None, None

    parts = text.split("|")
    declared_points = None
    if parts:
        try:
            declared_points = int(float(parts[0]))
        except (TypeError, ValueError):
            declared_points = None

    values = []
    for point in parts[1:]:
        if "," not in point:
            continue
        _, raw_value = point.split(",", 1)
        numeric_value = safe_float(raw_value)
        if numeric_value is not None and math.isfinite(numeric_value):
            values.append(numeric_value)

    if not values:
        return None, declared_points

    return np.asarray(values, dtype=float), declared_points


def normalize_wave(values: np.ndarray) -> np.ndarray:
    """对波形做稳健标准化，避免常量序列导致异常。"""
    signal = np.asarray(values, dtype=float)
    if signal.size == 0:
        return signal

    mean_value = np.nanmean(signal)
    centered = signal - mean_value
    std_value = np.nanstd(signal)

    if not math.isfinite(std_value) or std_value == 0:
        normalized = centered
    else:
        normalized = centered / std_value

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized


def compute_zero_crossings(signal: np.ndarray) -> int:
    """统计标准化信号的过零次数。"""
    if signal.size < 2:
        return 0
    signs = np.sign(signal)
    signs[signs == 0] = 1
    return int(np.sum(signs[1:] != signs[:-1]))


def extract_time_features(signal: np.ndarray, declared_points: int | None = None) -> dict[str, float]:
    """提取波形时域特征，并处理短序列、无峰值等边界情况。"""
    feats = default_wave_features()
    arr = np.asarray(signal, dtype=float)

    if arr.size == 0:
        feats["p4205_declared_points"] = declared_points
        return feats

    normalized = normalize_wave(arr)
    q75, q25 = np.percentile(normalized, [75, 25])
    diff1 = np.diff(normalized) if normalized.size > 1 else np.array([0.0])

    peak_distance = max(1, normalized.size // 20)
    peaks, _ = find_peaks(normalized, distance=peak_distance)

    feats.update(
        {
            "p4205_signal_length": float(arr.size),
            "p4205_declared_points": float(declared_points) if declared_points is not None else np.nan,
            "p4205_parse_ratio": (
                float(arr.size / declared_points) if declared_points not in (None, 0) else np.nan
            ),
            "p4205_mean": float(np.mean(normalized)),
            "p4205_std": float(np.std(normalized)),
            "p4205_max": float(np.max(normalized)),
            "p4205_min": float(np.min(normalized)),
            "p4205_median": float(np.median(normalized)),
            "p4205_skew": float(skew(normalized, bias=False)) if normalized.size >= 3 else 0.0,
            "p4205_kurtosis": float(kurtosis(normalized, bias=False)) if normalized.size >= 4 else 0.0,
            "p4205_range": float(np.max(normalized) - np.min(normalized)),
            "p4205_iqr": float(q75 - q25),
            "p4205_peak_count": float(len(peaks)),
            "p4205_max_rise": float(np.max(diff1)),
            "p4205_max_fall": float(np.min(diff1)),
            "p4205_area": float(np.trapezoid(normalized)),
            "p4205_energy": float(np.sum(normalized**2)),
            "p4205_zero_crossing_count": float(compute_zero_crossings(normalized)),
        }
    )

    if len(peaks) > 0:
        main_peak_index = peaks[np.argmax(normalized[peaks])]
        feats["p4205_main_peak_pos"] = float(main_peak_index)
        feats["p4205_main_peak_value"] = float(normalized[main_peak_index])
    else:
        feats["p4205_main_peak_pos"] = -1.0
        feats["p4205_main_peak_value"] = -1.0

    return feats


def extract_freq_features(signal: np.ndarray) -> dict[str, float]:
    """提取波形频域特征，并补充频谱能量与谱熵。"""
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        return {
            "p4205_fft_mean": np.nan,
            "p4205_fft_std": np.nan,
            "p4205_fft_max": np.nan,
            "p4205_dominant_freq": np.nan,
            "p4205_spectral_energy": np.nan,
            "p4205_spectral_entropy": np.nan,
        }

    normalized = normalize_wave(arr)
    fft_values = np.abs(fft(normalized))
    half_spectrum = fft_values[: max(1, len(fft_values) // 2)]

    power = half_spectrum**2
    total_power = float(np.sum(power))
    if total_power > 0:
        power_ratio = power / total_power
        spectral_entropy = float(-np.sum(power_ratio * np.log(power_ratio + 1e-12)))
    else:
        spectral_entropy = 0.0

    return {
        "p4205_fft_mean": float(np.mean(half_spectrum)),
        "p4205_fft_std": float(np.std(half_spectrum)),
        "p4205_fft_max": float(np.max(half_spectrum)),
        "p4205_dominant_freq": float(np.argmax(half_spectrum)),
        "p4205_spectral_energy": total_power,
        "p4205_spectral_entropy": spectral_entropy,
    }


def build_p4205_feature_frame(series: pd.Series) -> pd.DataFrame:
    """批量处理 `p4205_i0` 列并返回对齐的波形特征表。"""
    feature_rows = []
    for wave_str in series.tolist():
        signal, declared_points = parse_waveform(wave_str)
        if signal is None or signal.size == 0:
            feature_rows.append(default_wave_features())
            continue

        time_features = extract_time_features(signal, declared_points=declared_points)
        freq_features = extract_freq_features(signal)
        feature_rows.append({**time_features, **freq_features})

    feature_frame = pd.DataFrame(feature_rows, index=series.index)
    return feature_frame.reindex(columns=WAVE_FEATURE_COLUMNS)
