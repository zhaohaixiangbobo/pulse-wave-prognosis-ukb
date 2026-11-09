"""
文件说明：训练 10 年全因死亡风险 Torch 二分类模型，并优先优化 PR-AUC。

本脚本使用最新标签定义：`target=基线后 target-years 年内是否死亡`。相比旧版实现，
新增双通道 p4205_i0 波形输入、二分类 BCE/Focal 损失、gated 双塔融合、PR-AUC
早停策略和 Torch late fusion 强基线。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader, Dataset

from p4205_feature_policy import drop_p4205_derived_structured_features
from p4205_advanced_features import ADVANCED_FEATURE_COLUMNS, build_advanced_feature_frame
from pulse_wave_features import WAVE_FEATURE_COLUMNS, build_p4205_feature_frame
from two_tower_torch_classifier import (
    BinaryGatedThreeTowerClassifier,
    BinaryGatedTwoTowerClassifier,
    BinaryTabularOnlyClassifier,
    BinaryThreeTowerClassifier,
    BinaryTwoTowerClassifier,
    BinaryWaveOnlyClassifier,
)
from two_tower_torch_model import RANDOM_STATE, WaveformConfig, parse_p4205_i0_to_tensor, set_global_seed


EID_COLUMN = "eid"
WAVE_COLUMN = "p4205_i0"
AGE_COLUMN = "p21022"
DEATH_AGE_COLUMN = "p40007_i0"
DEFAULT_DATA = Path("data/data.csv")
DEFAULT_OUTPUT = Path("data/model_output_torch_10year_cls")
DEFAULT_P4205_CACHE = Path("data/p4205_advanced_features.csv")
LATE_FUSION_NAME = "torch_late_fusion"
STACKING_FUSION_NAME = "torch_stacking_fusion"

BASELINE_COLUMNS = [
    "p21021_i0",  # 脉冲波动脉刚度指数
    "p4199_i0",  # 脉冲波缺口位置
    "p4200_i0",  # 波形肩部位置
    AGE_COLUMN,
    "p20160_i0",  # 是否吸过烟
    "p20162_i0",  # 吸烟暴露比例
    "p3616_i0",  # 行走胸痛停止
    "p3751_i0",  # 上坡或匆忙时胸痛
    "p4717_i0",  # 平地行走呼吸急促
    "p2316_i0",  # 胸口喘息
    "sex",
]
SYMPTOM_COLUMNS = ["p3616_i0", "p3751_i0", "p4717_i0", "p2316_i0"]
READ_COLUMNS = [EID_COLUMN, WAVE_COLUMN, DEATH_AGE_COLUMN, *BASELINE_COLUMNS]
MODEL_CHOICES = [
    "tabular_only",
    "wave_only_cnn",
    "wave_only_tcn",
    "wave_only_transformer",
    "p4205_feature_only_mlp",
    "two_tower_cnn",
    "two_tower_tcn",
    "two_tower_cnn_gated",
    "two_tower_tcn_gated",
    "three_tower_cnn",
    "three_tower_cnn_gated",
    "three_tower_transformer_gated",
    "three_tower_hybrid_gated",
    "tabular_plus_p4205_cross_mlp",
]
COHORT_CHOICES = [
    "all_p4205_valid",
    "quality_strict",
    "age_45_70",
    "complete_core_features",
    "high_stiffness_or_smoking",
]


class TwoTowerBinaryDataset(Dataset):
    """二分类双塔数据集：p4205_i0 字符串 + 结构化特征向量 -> 0/1 标签。"""

    def __init__(
        self,
        eid_values: np.ndarray,
        wave_values: list[object],
        tabular_features: np.ndarray,
        labels: np.ndarray,
        waveform_config: WaveformConfig,
    ) -> None:
        if len(wave_values) != tabular_features.shape[0] or tabular_features.shape[0] != labels.shape[0]:
            raise ValueError("wave/tabular/label 的样本数不一致。")
        self.eid_values = np.asarray(eid_values)
        self.wave_values = wave_values
        self.tabular_features = tabular_features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.waveform_config = waveform_config

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wave = parse_p4205_i0_to_tensor(self.wave_values[idx], config=self.waveform_config)
        tabular = torch.as_tensor(self.tabular_features[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        return wave, tabular, label


class ThreeTowerBinaryDataset(Dataset):
    """二分类三塔数据集：原始波形、p4205 人工特征和基线结构化特征并行输入。"""

    def __init__(
        self,
        eid_values: np.ndarray,
        wave_values: list[object],
        p4205_features: np.ndarray,
        tabular_features: np.ndarray,
        labels: np.ndarray,
        waveform_config: WaveformConfig,
    ) -> None:
        if (
            len(wave_values) != p4205_features.shape[0]
            or p4205_features.shape[0] != tabular_features.shape[0]
            or tabular_features.shape[0] != labels.shape[0]
        ):
            raise ValueError("wave/p4205_feature/tabular/label 的样本数不一致。")
        self.eid_values = np.asarray(eid_values)
        self.wave_values = wave_values
        self.p4205_features = p4205_features.astype(np.float32, copy=False)
        self.tabular_features = tabular_features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.waveform_config = waveform_config

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wave = parse_p4205_i0_to_tensor(self.wave_values[idx], config=self.waveform_config)
        p4205_feature = torch.as_tensor(self.p4205_features[idx], dtype=torch.float32)
        tabular = torch.as_tensor(self.tabular_features[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        return wave, p4205_feature, tabular, label


class WaveOnlyBinaryDataset(Dataset):
    """二分类波形单塔数据集：仅使用 p4205_i0 双通道波形。"""

    def __init__(
        self,
        eid_values: np.ndarray,
        wave_values: list[object],
        labels: np.ndarray,
        waveform_config: WaveformConfig,
    ) -> None:
        if len(wave_values) != labels.shape[0]:
            raise ValueError("wave/label 的样本数不一致。")
        self.eid_values = np.asarray(eid_values)
        self.wave_values = wave_values
        self.labels = labels.astype(np.float32, copy=False)
        self.waveform_config = waveform_config

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wave = parse_p4205_i0_to_tensor(self.wave_values[idx], config=self.waveform_config)
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        return wave, label


class TabularOnlyBinaryDataset(Dataset):
    """二分类结构化单塔数据集：仅使用增强后的基线结构化特征。"""

    def __init__(self, eid_values: np.ndarray, tabular_features: np.ndarray, labels: np.ndarray) -> None:
        if tabular_features.shape[0] != labels.shape[0]:
            raise ValueError("tabular/label 的样本数不一致。")
        self.eid_values = np.asarray(eid_values)
        self.tabular_features = tabular_features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tabular = torch.as_tensor(self.tabular_features[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        return tabular, label


class FocalBCEWithLogitsLoss(nn.Module):
    """二分类 Focal BCE，适合低阳性率风险预测任务。"""

    def __init__(self, alpha: float, gamma: float, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probability = torch.sigmoid(logits)
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


class WeightedCrossEntropyFromBinaryLogit(nn.Module):
    """将 `[B]` 二分类 logit 转成两类 logits 后计算加权 CE，保留对照损失选项。"""

    def __init__(self, class_weight: torch.Tensor) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        two_class_logits = torch.stack([-0.5 * logits, 0.5 * logits], dim=1)
        return self.loss(two_class_logits, target.long())


def resolve_output_path(path_value: str | Path) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return (Path(__file__).resolve().parent.parent / "data" / raw).resolve()


def load_data(path: Path, target_years: float, max_rows: int | None) -> pd.DataFrame:
    """日期版只读取冻结十年标签，禁止以年龄差覆盖目标。"""
    from exact_data import load_exact_data
    if target_years != 10.0:
        raise ValueError('日期版主目标固定十年')
    return load_exact_data(path, max_rows)


def apply_cohort_filter(df: pd.DataFrame, cohort: str, p4205_cache: Path | None) -> pd.DataFrame:
    """按预设 cohort 重新筛选样本，仅使用基线可得字段和 p4205 质量字段。"""
    filtered = df.copy()
    if cohort == "all_p4205_valid":
        return filtered.reset_index(drop=True)
    if cohort == "age_45_70":
        age = pd.to_numeric(filtered[AGE_COLUMN], errors="coerce")
        return filtered[age.between(45, 70, inclusive="both")].reset_index(drop=True)
    if cohort == "complete_core_features":
        core = ["p21021_i0", "p4199_i0", "p4200_i0", "p20162_i0"]
        return filtered.dropna(subset=core).reset_index(drop=True)
    if cohort == "high_stiffness_or_smoking":
        stiffness = pd.to_numeric(filtered["p21021_i0"], errors="coerce")
        smoking = pd.to_numeric(filtered["p20162_i0"], errors="coerce")
        threshold = float(stiffness.median())
        return filtered[stiffness.ge(threshold) | smoking.ge(0.5)].reset_index(drop=True)
    if cohort == "quality_strict":
        if p4205_cache is None or not p4205_cache.exists():
            raise FileNotFoundError("quality_strict cohort 需要可用的 p4205 高级特征缓存。")
        quality_columns = [
            EID_COLUMN,
            "p4205_q_parse_success",
            "p4205_q_expected_length",
            "p4205_q_nonflat",
            "p4205_q_h1_success",
            "p4205_q_h4_success",
            "p4205_q_width_success",
        ]
        quality = pd.read_csv(p4205_cache, usecols=lambda column: column in quality_columns)
        quality = quality.drop_duplicates(EID_COLUMN).set_index(EID_COLUMN)
        aligned = quality.reindex(filtered[EID_COLUMN].to_numpy())
        keep = aligned.drop(columns=[], errors="ignore").eq(1.0).all(axis=1).to_numpy()
        return filtered[keep].reset_index(drop=True)
    raise ValueError(f"未知 cohort：{cohort}")


def yes_indicator(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.eq("yes").astype(np.float32)


def build_structured_frame(df: pd.DataFrame, feature_set: str = "current") -> pd.DataFrame:
    """构建基线结构化特征，避免使用 i2 随访和死亡结局字段。"""
    frame = df[BASELINE_COLUMNS].copy()
    numeric_columns = ["p21021_i0", "p4199_i0", "p4200_i0", AGE_COLUMN, "p20162_i0"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    age = frame[AGE_COLUMN]
    stiffness = frame["p21021_i0"]
    notch = frame["p4199_i0"]
    shoulder = frame["p4200_i0"]

    # 年龄、动脉刚度、吸烟暴露和波形位置是最新树模型中最稳定的主信号。
    frame["age_squared"] = age**2
    frame["stiffness_x_age"] = stiffness * age
    frame["stiffness_per_age"] = stiffness / age.replace(0, np.nan)
    frame["notch_shoulder_gap"] = notch - shoulder
    frame["notch_shoulder_midpoint"] = (notch + shoulder) / 2.0
    frame["notch_position_ratio"] = notch / 100.0
    frame["shoulder_position_ratio"] = shoulder / 100.0
    frame["smoking_exposure_x_age"] = frame["p20162_i0"] * age
    frame["symptom_yes_count"] = sum(yes_indicator(df[column]) for column in SYMPTOM_COLUMNS)
    frame["baseline_missing_count"] = df[BASELINE_COLUMNS].isna().sum(axis=1).astype(np.float32)
    # clinical_no_p4205 用于严格检验 raw/engineered P4205 是否提供新增信息。
    frame, _ = drop_p4205_derived_structured_features(frame, feature_set)
    from exact_data import apply_variant
    frame = apply_variant(frame, df)
    return frame.replace([np.inf, -np.inf], np.nan)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_tabular_preprocessor(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_columns, categorical_columns


def _parse_wave_values_for_features(value: object) -> np.ndarray:
    if not isinstance(value, str) or "|" not in value:
        return np.asarray([], dtype=np.float32)
    values: list[float] = []
    for point in value.strip().split("|")[1:]:
        if "," not in point:
            continue
        _, raw_value = point.split(",", 1)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            values.append(numeric_value)
    return np.asarray(values, dtype=np.float32)


def build_raw_diff_stat_frame(series: pd.Series) -> pd.DataFrame:
    """补充 raw/diff 统计特征，显式暴露波形一阶变化强度和尾部形态。"""
    rows: list[dict[str, float]] = []
    for value in series.tolist():
        raw = _parse_wave_values_for_features(value)
        if raw.size == 0:
            rows.append(
                {
                    "p4205_raw_point_mean": np.nan,
                    "p4205_raw_point_std": np.nan,
                    "p4205_raw_point_min": np.nan,
                    "p4205_raw_point_max": np.nan,
                    "p4205_raw_point_range": np.nan,
                    "p4205_diff_mean": np.nan,
                    "p4205_diff_std": np.nan,
                    "p4205_diff_abs_mean": np.nan,
                    "p4205_diff_abs_max": np.nan,
                    "p4205_diff_positive_ratio": np.nan,
                }
            )
            continue
        diff = np.diff(raw) if raw.size > 1 else np.asarray([0.0], dtype=np.float32)
        rows.append(
            {
                "p4205_raw_point_mean": float(np.mean(raw)),
                "p4205_raw_point_std": float(np.std(raw)),
                "p4205_raw_point_min": float(np.min(raw)),
                "p4205_raw_point_max": float(np.max(raw)),
                "p4205_raw_point_range": float(np.max(raw) - np.min(raw)),
                "p4205_diff_mean": float(np.mean(diff)),
                "p4205_diff_std": float(np.std(diff)),
                "p4205_diff_abs_mean": float(np.mean(np.abs(diff))),
                "p4205_diff_abs_max": float(np.max(np.abs(diff))),
                "p4205_diff_positive_ratio": float(np.mean(diff > 0)),
            }
        )
    return pd.DataFrame(rows, index=series.index)


def load_cached_advanced_feature_frame(cache_path: Path, expected_eids: pd.Series) -> pd.DataFrame | None:
    """按 eid 读取已缓存的 p4205 高级特征，避免全量运行时重复做昂贵特征提取。"""
    if not cache_path.exists():
        return None
    columns = [EID_COLUMN, *ADVANCED_FEATURE_COLUMNS]
    cached = pd.read_csv(cache_path, usecols=lambda column: column in columns)
    if EID_COLUMN not in cached.columns:
        return None
    # landmark 日期合并会把 EID 规范为 pandas string；缓存 CSV 常被推断为整数。
    # 在 reindex 前统一为字符串，避免类型不同导致所有高级 P4205 特征误对齐为缺失。
    expected = expected_eids.astype("string")
    cached[EID_COLUMN] = cached[EID_COLUMN].astype("string")
    cached = cached.drop_duplicates(subset=[EID_COLUMN], keep="first").set_index(EID_COLUMN)
    aligned = cached.reindex(expected.to_numpy())
    aligned.index = expected_eids.index
    return aligned.reindex(columns=ADVANCED_FEATURE_COLUMNS)


def build_p4205_feature_tower_frame(
    series: pd.Series,
    eid_values: pd.Series | None = None,
    cache_path: Path | None = None,
    include_nonlinear: bool = True,
) -> pd.DataFrame:
    """构建 p4205 feature tower 输入：已有 summary、高级五大域特征和 raw/diff 统计。"""
    summary = build_p4205_feature_frame(series)
    advanced = None
    if cache_path is not None and eid_values is not None:
        advanced = load_cached_advanced_feature_frame(cache_path, eid_values)
    if advanced is None:
        advanced = build_advanced_feature_frame(series, include_nonlinear=include_nonlinear)
    raw_diff = build_raw_diff_stat_frame(series)
    frame = pd.concat([summary, advanced, raw_diff], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()].replace([np.inf, -np.inf], np.nan)
    return frame


def build_numeric_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = feature_df.columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        ],
        remainder="drop",
    )


def make_oof_p4205_risk(
    p4205_train: np.ndarray,
    y_train: np.ndarray,
    p4205_val: np.ndarray,
    p4205_test: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """训练集内 OOF 生成 p4205 风险分数，避免交叉特征直接泄漏训练标签。"""
    from sklearn.model_selection import StratifiedKFold

    class_counts = np.bincount(y_train, minlength=2)
    if int(class_counts.min()) < 2:
        fallback = float(np.mean(y_train))
        return (
            np.full(p4205_train.shape[0], fallback, dtype=np.float32),
            np.full(p4205_val.shape[0], fallback, dtype=np.float32),
            np.full(p4205_test.shape[0], fallback, dtype=np.float32),
        )
    splits = max(2, min(int(n_splits), int(class_counts.min())))
    oof = np.zeros(p4205_train.shape[0], dtype=np.float32)
    val_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    for fold, (fit_idx, hold_idx) in enumerate(cv.split(p4205_train, y_train), start=1):
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE + fold)
        model.fit(p4205_train[fit_idx], y_train[fit_idx])
        oof[hold_idx] = model.predict_proba(p4205_train[hold_idx])[:, 1].astype(np.float32)
        val_predictions.append(model.predict_proba(p4205_val)[:, 1].astype(np.float32))
        test_predictions.append(model.predict_proba(p4205_test)[:, 1].astype(np.float32))
    return oof, np.mean(val_predictions, axis=0), np.mean(test_predictions, axis=0)


def build_cross_feature_frame(structured: pd.DataFrame, p4205_risk: np.ndarray) -> pd.DataFrame:
    """把 p4205 风险分数与关键临床/波形字段显式交叉，表达人群异质性。"""
    frame = structured.copy()
    risk = pd.Series(p4205_risk, index=frame.index, name="p4205_oof_risk").astype(float)
    numeric_inputs = {
        "age": pd.to_numeric(frame.get(AGE_COLUMN), errors="coerce"),
        "stiffness": pd.to_numeric(frame.get("p21021_i0"), errors="coerce"),
        "notch": pd.to_numeric(frame.get("p4199_i0"), errors="coerce"),
        "shoulder": pd.to_numeric(frame.get("p4200_i0"), errors="coerce"),
        "smoking_exposure": pd.to_numeric(frame.get("p20162_i0"), errors="coerce"),
        "symptom_yes_count": pd.to_numeric(frame.get("symptom_yes_count"), errors="coerce"),
    }
    frame["p4205_oof_risk"] = risk
    for name, values in numeric_inputs.items():
        frame[f"{name}_x_p4205_oof_risk"] = values * risk
    frame["notch_shoulder_gap_x_p4205_oof_risk"] = pd.to_numeric(
        frame.get("notch_shoulder_gap"), errors="coerce"
    ) * risk
    return frame.replace([np.inf, -np.inf], np.nan)


def build_stacking_meta_frame(
    probability_sources: dict[str, np.ndarray],
    structured_subset: pd.DataFrame,
) -> pd.DataFrame:
    """构建验证集/测试集 stacking 输入，meta model 只在验证集拟合。"""
    frame = pd.DataFrame({f"{name}_probability": value for name, value in probability_sources.items()})
    if {"tabular", "wave"}.issubset(probability_sources):
        frame["tabular_x_wave_probability"] = probability_sources["tabular"] * probability_sources["wave"]
    if {"tabular", "three_tower"}.issubset(probability_sources):
        frame["tabular_x_three_tower_probability"] = (
            probability_sources["tabular"] * probability_sources["three_tower"]
        )
    age = pd.to_numeric(structured_subset[AGE_COLUMN], errors="coerce").to_numpy()
    stiffness = pd.to_numeric(structured_subset["p21021_i0"], errors="coerce").to_numpy()
    smoking = pd.to_numeric(structured_subset["p20162_i0"], errors="coerce").to_numpy()
    frame["age"] = age
    frame["p21021_i0"] = stiffness
    frame["p4199_i0"] = pd.to_numeric(structured_subset["p4199_i0"], errors="coerce").to_numpy()
    frame["p4200_i0"] = pd.to_numeric(structured_subset["p4200_i0"], errors="coerce").to_numpy()
    frame["smoking_exposure"] = smoking
    frame["symptom_yes_count"] = pd.to_numeric(structured_subset["symptom_yes_count"], errors="coerce").to_numpy()
    for name, probability in probability_sources.items():
        frame[f"{name}_probability_x_age"] = probability * age
        frame[f"{name}_probability_x_stiffness"] = probability * stiffness
        frame[f"{name}_probability_x_smoking"] = probability * smoking
    return frame.replace([np.inf, -np.inf], np.nan)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def move_batch_to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=False) for item in batch)


def model_forward(model: nn.Module, batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch = move_batch_to_device(batch, device)
    if isinstance(model, (BinaryThreeTowerClassifier, BinaryGatedThreeTowerClassifier)):
        wave_x, p4205_x, tabular_x, y = batch
        return model(wave_x, p4205_x, tabular_x), y
    if isinstance(model, (BinaryTwoTowerClassifier, BinaryGatedTwoTowerClassifier)):
        wave_x, tabular_x, y = batch
        return model(wave_x, tabular_x), y
    if isinstance(model, BinaryWaveOnlyClassifier):
        wave_x, y = batch
        return model(wave_x), y
    if isinstance(model, BinaryTabularOnlyClassifier):
        tabular_x, y = batch
        return model(tabular_x), y
    raise TypeError(f"不支持的模型类型：{type(model)}")


def class_weight_tensors(labels: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    negative = int((labels == 0).sum())
    positive = int((labels == 1).sum())
    total = max(1, negative + positive)
    ce_weight = torch.as_tensor(
        [total / max(1, 2 * negative), total / max(1, 2 * positive)],
        dtype=torch.float32,
        device=device,
    )
    pos_weight = torch.as_tensor(float(negative / max(1, positive)), dtype=torch.float32, device=device)
    return ce_weight, pos_weight


def make_loss(
    loss_name: str,
    labels: np.ndarray,
    device: torch.device,
    focal_alpha: float,
    focal_gamma: float,
) -> nn.Module:
    """根据训练集标签构建损失函数，避免使用验证/测试集信息。"""
    ce_weight, pos_weight = class_weight_tensors(labels, device)
    if loss_name == "focal_bce":
        return FocalBCEWithLogitsLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
    if loss_name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == "weighted_ce":
        return WeightedCrossEntropyFromBinaryLogit(ce_weight)
    raise ValueError(f"未知损失函数：{loss_name}")


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        logits, y = model_forward(model, batch, device)
        loss = criterion(logits, y)
        probability = torch.sigmoid(logits)
        losses.append(float(loss.detach().cpu().item()))
        labels.append(y.detach().cpu().numpy().astype(np.int64))
        probabilities.append(probability.detach().cpu().numpy())
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_true = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)
    y_probability = np.concatenate(probabilities) if probabilities else np.empty((0,), dtype=np.float32)
    return mean_loss, y_true, y_probability


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """在验证集上选择平衡准确率最高的筛查阈值。"""
    best_score = -np.inf
    best_threshold = 0.5
    for threshold in np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 97))):
        score = balanced_accuracy_score(y_true, probability >= threshold)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, object]:
    prediction = (probability >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "precision_positive": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    metrics.update(topk_metrics(y_true, probability))
    return metrics


def topk_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    percents: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
) -> dict[str, float]:
    """计算高风险 Top-K precision/recall，用于评估筛查队列质量。"""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    positive_count = int(y.sum())
    result: dict[str, float] = {}
    for percent in percents:
        label = int(round(percent * 100))
        if y.size == 0:
            result[f"top_{label}pct_precision"] = float("nan")
            result[f"top_{label}pct_recall"] = float("nan")
            continue
        k = max(1, int(np.ceil(y.size * percent)))
        order = np.argsort(-p, kind="mergesort")
        top_y = y[order[:k]]
        result[f"top_{label}pct_precision"] = float(top_y.mean())
        result[f"top_{label}pct_recall"] = float(top_y.sum() / positive_count) if positive_count > 0 else 0.0
    return result


def metric_for_early_stop(metrics: dict[str, object], metric_name: str) -> float:
    if metric_name == "pr_auc":
        return float(metrics["pr_auc"])
    if metric_name == "roc_auc":
        return float(metrics["roc_auc"])
    if metric_name == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    raise ValueError(f"不支持的 early stop 指标：{metric_name}")


def save_predictions(path: Path, eid_values: np.ndarray, y_true: np.ndarray, probability: np.ndarray, threshold: float) -> None:
    prediction = (probability >= threshold).astype("int8")
    pd.DataFrame(
        {
            EID_COLUMN: eid_values,
            "target_true": y_true,
            "probability_10year_death": probability,
            "prediction": prediction,
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def save_loss_curve(history_df: pd.DataFrame, output_path: Path, model_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_one_model(
    model: nn.Module,
    model_name: str,
    output_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    val_eids: np.ndarray,
    test_eids: np.ndarray,
    train_labels: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    patience: int,
    device: torch.device,
    model_config: dict[str, object],
    loss_name: str,
    focal_alpha: float,
    focal_gamma: float,
    early_stop_metric: str,
    min_delta: float,
) -> dict[str, object]:
    criterion = make_loss(loss_name, train_labels, device, focal_alpha, focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -np.inf
    best_val_metrics: dict[str, object] | None = None
    no_improve = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, y = model_forward(model, batch, device)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss, y_val, val_probability = collect_predictions(model, val_loader, criterion, device)
        threshold, _ = select_threshold(y_val, val_probability)
        val_metrics = binary_metrics(y_val, val_probability, threshold)
        current_score = metric_for_early_stop(val_metrics, early_stop_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_threshold": threshold,
            }
        )
        print(
            f"[{model_name}] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_auc={val_metrics['roc_auc']:.6f} val_pr={val_metrics['pr_auc']:.6f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.6f}"
        )

        if current_score > best_score + min_delta:
            best_score = current_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            no_improve = 0
        else:
            no_improve += 1
        if patience > 0 and no_improve >= patience:
            print(f"[{model_name}] early stop at epoch={epoch} (patience={patience})")
            break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError(f"{model_name} 未产生有效模型。")

    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "loss_history.csv", index=False, encoding="utf-8-sig")
    model.load_state_dict(best_state)

    val_loss, y_val, val_probability = collect_predictions(model, val_loader, criterion, device)
    threshold, validation_balanced_accuracy = select_threshold(y_val, val_probability)
    test_loss, y_test, test_probability = collect_predictions(model, test_loader, criterion, device)
    test_metrics = binary_metrics(y_test, test_probability, threshold)
    payload = {
        "model_name": model_name,
        "model_config": model_config,
        "loss_name": loss_name,
        "early_stop_metric": early_stop_metric,
        "best_epoch": int(best_epoch),
        "best_val_epoch": int(best_epoch),
        "best_val_pr_auc": float(best_val_metrics["pr_auc"]),
        "best_val_roc_auc": float(best_val_metrics["roc_auc"]),
        "best_val_balanced_accuracy": float(best_val_metrics["balanced_accuracy"]),
        "validation_loss": float(val_loss),
        "test_loss": float(test_loss),
        "validation_balanced_accuracy_at_threshold": float(validation_balanced_accuracy),
        "test": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "model_config.json").write_text(json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8")
    save_predictions(output_dir / "validation_predictions.csv", val_eids, y_val, val_probability, threshold)
    save_predictions(output_dir / "test_predictions.csv", test_eids, y_test, test_probability, threshold)
    save_loss_curve(history_df, output_dir / "loss_curve.png", model_name)
    torch.save(best_state, output_dir / "model_state.pt")
    payload["_validation_probability"] = val_probability
    payload["_validation_true"] = y_val
    payload["_test_probability"] = test_probability
    payload["_test_true"] = y_test
    return payload


def select_late_fusion(
    y_val: np.ndarray,
    tabular_probability: np.ndarray,
    wave_probability: np.ndarray,
    max_weight: float,
    step: float,
) -> tuple[float, float, float]:
    """仅使用验证集选择 late fusion 权重和阈值，不读取测试集标签。"""
    best_weight = 0.0
    best_roc_auc = -np.inf
    best_threshold = 0.5
    weights = np.arange(0.0, max_weight + step / 2.0, step)
    for weight in weights:
        probability = (1.0 - weight) * tabular_probability + weight * wave_probability
        roc_auc = float(roc_auc_score(y_val, probability))
        threshold, _ = select_threshold(y_val, probability)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_weight = float(weight)
            best_threshold = float(threshold)
    return best_weight, best_threshold, best_roc_auc


def export_late_fusion(
    base_output: Path,
    tabular_result: dict[str, object],
    wave_result: dict[str, object],
    val_eids: np.ndarray,
    test_eids: np.ndarray,
    max_weight: float,
    step: float,
) -> dict[str, object]:
    y_val = np.asarray(tabular_result["_validation_true"], dtype=np.int64)
    y_test = np.asarray(tabular_result["_test_true"], dtype=np.int64)
    tab_val = np.asarray(tabular_result["_validation_probability"], dtype=np.float32)
    wave_val = np.asarray(wave_result["_validation_probability"], dtype=np.float32)
    tab_test = np.asarray(tabular_result["_test_probability"], dtype=np.float32)
    wave_test = np.asarray(wave_result["_test_probability"], dtype=np.float32)

    weight, threshold, validation_roc_auc = select_late_fusion(y_val, tab_val, wave_val, max_weight, step)
    val_probability = (1.0 - weight) * tab_val + weight * wave_val
    test_probability = (1.0 - weight) * tab_test + weight * wave_test
    payload = {
        "model_name": LATE_FUSION_NAME,
        "source_models": ["tabular_only", "wave_only_cnn"],
        "selected_wave_weight": float(weight),
        "selected_tabular_weight": float(1.0 - weight),
        "validation_roc_auc": float(validation_roc_auc),
        "validation_pr_auc": float(average_precision_score(y_val, val_probability)),
        "validation": binary_metrics(y_val, val_probability, threshold),
        "test": binary_metrics(y_test, test_probability, threshold),
    }
    output_dir = base_output / LATE_FUSION_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_predictions(output_dir / "validation_predictions.csv", val_eids, y_val, val_probability, threshold)
    save_predictions(output_dir / "test_predictions.csv", test_eids, y_test, test_probability, threshold)
    return payload


def export_stacking_fusion(
    base_output: Path,
    model_payloads: dict[str, dict[str, object]],
    structured: pd.DataFrame,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    val_eids: np.ndarray,
    test_eids: np.ndarray,
) -> dict[str, object] | None:
    """用验证集训练 LogisticRegression meta model，测试集只做最终评估。"""
    required = ["tabular_only", "wave_only_cnn"]
    if not all(name in model_payloads for name in required):
        return None
    three_tower_name = next(
        (
            name
            for name in [
                "three_tower_hybrid_gated",
                "three_tower_transformer_gated",
                "three_tower_cnn_gated",
                "two_tower_cnn_gated",
                "two_tower_cnn",
            ]
            if name in model_payloads
        ),
        None,
    )

    tabular_result = model_payloads["tabular_only"]
    wave_result = model_payloads["wave_only_cnn"]
    y_val = np.asarray(tabular_result["_validation_true"], dtype=np.int64)
    y_test = np.asarray(tabular_result["_test_true"], dtype=np.int64)

    val_sources = {
        "tabular": np.asarray(tabular_result["_validation_probability"], dtype=np.float32),
        "wave": np.asarray(wave_result["_validation_probability"], dtype=np.float32),
    }
    test_sources = {
        "tabular": np.asarray(tabular_result["_test_probability"], dtype=np.float32),
        "wave": np.asarray(wave_result["_test_probability"], dtype=np.float32),
    }
    if three_tower_name is not None:
        val_sources["three_tower"] = np.asarray(model_payloads[three_tower_name]["_validation_probability"], dtype=np.float32)
        test_sources["three_tower"] = np.asarray(model_payloads[three_tower_name]["_test_probability"], dtype=np.float32)
    if "p4205_feature_only_mlp" in model_payloads:
        val_sources["p4205_feature"] = np.asarray(
            model_payloads["p4205_feature_only_mlp"]["_validation_probability"], dtype=np.float32
        )
        test_sources["p4205_feature"] = np.asarray(
            model_payloads["p4205_feature_only_mlp"]["_test_probability"], dtype=np.float32
        )

    val_frame = build_stacking_meta_frame(val_sources, structured.iloc[val_idx])
    test_frame = build_stacking_meta_frame(test_sources, structured.iloc[test_idx])
    tabular_top5 = float(topk_metrics(y_val, val_sources["tabular"])["top_5pct_precision"])
    candidates = {
        "logistic_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("meta", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "calibrated_logistic_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "meta",
                    CalibratedClassifierCV(
                        estimator=LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
                        cv=3,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_depth2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "meta",
                    HistGradientBoostingClassifier(
                        max_iter=100,
                        max_leaf_nodes=7,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    candidate_rows: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}
    for name, model in candidates.items():
        try:
            model.fit(val_frame, y_val)
            probability = model.predict_proba(val_frame)[:, 1].astype(np.float32)
            threshold_candidate, _ = select_threshold(y_val, probability)
            metrics = binary_metrics(y_val, probability, threshold_candidate)
            candidate_rows.append(
                {
                    "name": name,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "top_5pct_precision": metrics["top_5pct_precision"],
                    "passes_top5_guardrail": bool(float(metrics["top_5pct_precision"]) >= tabular_top5),
                    "fit_error": None,
                }
            )
            fitted[name] = model
        except Exception as exc:
            candidate_rows.append(
                {
                    "name": name,
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "top_5pct_precision": float("nan"),
                    "passes_top5_guardrail": False,
                    "fit_error": repr(exc),
                }
            )
    if not fitted:
        return None
    candidate_frame = pd.DataFrame([row for row in candidate_rows if row["name"] in fitted]).sort_values(
        ["passes_top5_guardrail", "roc_auc", "pr_auc"], ascending=[False, False, False]
    )
    selected_name = str(candidate_frame.iloc[0]["name"])
    selected_model = fitted[selected_name]
    val_probability = selected_model.predict_proba(val_frame)[:, 1].astype(np.float32)
    test_probability = selected_model.predict_proba(test_frame)[:, 1].astype(np.float32)
    threshold, _ = select_threshold(y_val, val_probability)
    payload = {
        "model_name": STACKING_FUSION_NAME,
        "source_models": list(val_sources.keys()),
        "meta_model": selected_name,
        "meta_train_split": "validation_only",
        "meta_candidates": candidate_rows,
        "passes_top5_guardrail": bool(candidate_frame.iloc[0]["passes_top5_guardrail"]),
        "validation": binary_metrics(y_val, val_probability, threshold),
        "validation_roc_auc": float(roc_auc_score(y_val, val_probability)),
        "validation_pr_auc": float(average_precision_score(y_val, val_probability)),
        "test": binary_metrics(y_test, test_probability, threshold),
    }
    output_dir = base_output / STACKING_FUSION_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_predictions(output_dir / "validation_predictions.csv", val_eids, y_val, val_probability, threshold)
    save_predictions(output_dir / "test_predictions.csv", test_eids, y_test, test_probability, threshold)
    return payload


def public_metric_payload(payload: dict[str, object]) -> dict[str, object]:
    """移除内存中的概率数组，得到可序列化的公开指标。"""
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def aggregate_metrics(
    base_output: Path,
    model_payloads: dict[str, dict[str, object]],
    roc_auc_baseline: float,
    min_roc_gain: float,
    pr_guardrail_drop: float,
) -> None:
    rows: list[dict[str, object]] = []
    for model_name, payload in model_payloads.items():
        test = payload.get("test", {})
        rows.append(
            {
                "model_name": model_name,
                "best_epoch": payload.get("best_epoch"),
                "best_val_pr_auc": payload.get("best_val_pr_auc", payload.get("validation_pr_auc")),
                "best_val_roc_auc": payload.get("best_val_roc_auc", payload.get("validation_roc_auc")),
                "roc_auc": test.get("roc_auc"),
                "pr_auc": test.get("pr_auc"),
                "balanced_accuracy": test.get("balanced_accuracy"),
                "macro_f1": test.get("macro_f1"),
                "precision_positive": test.get("precision_positive"),
                "recall_sensitivity": test.get("recall_sensitivity"),
                "specificity": test.get("specificity"),
                "threshold": test.get("threshold"),
                "top_1pct_precision": test.get("top_1pct_precision"),
                "top_1pct_recall": test.get("top_1pct_recall"),
                "top_2pct_precision": test.get("top_2pct_precision"),
                "top_2pct_recall": test.get("top_2pct_recall"),
                "top_5pct_precision": test.get("top_5pct_precision"),
                "top_5pct_recall": test.get("top_5pct_recall"),
                "top_10pct_precision": test.get("top_10pct_precision"),
                "top_10pct_recall": test.get("top_10pct_recall"),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        summary.to_csv(base_output / "metrics_summary.csv", index=False, encoding="utf-8-sig")
        (base_output / "metrics_summary.json").write_text("[]", encoding="utf-8")
        return

    tabular_row = summary[summary["model_name"] == "tabular_only"]
    if tabular_row.empty:
        reference = summary.sort_values("roc_auc", ascending=False).iloc[0]
    else:
        reference = tabular_row.iloc[0]

    tabular_roc = float(reference["roc_auc"])
    tabular_pr = float(reference["pr_auc"])
    tabular_top1 = float(reference["top_1pct_precision"])
    tabular_top5 = float(reference["top_5pct_precision"])

    summary["roc_auc_gain_vs_tabular"] = summary["roc_auc"].astype(float) - tabular_roc
    summary["pr_auc_gain_vs_tabular"] = summary["pr_auc"].astype(float) - tabular_pr
    summary["top_1pct_precision_gain_vs_tabular"] = summary["top_1pct_precision"].astype(float) - tabular_top1
    summary["top_5pct_precision_gain_vs_tabular"] = summary["top_5pct_precision"].astype(float) - tabular_top5
    summary["passes_roc_auc_threshold"] = summary["roc_auc"].astype(float) >= float(roc_auc_baseline + min_roc_gain)
    summary["passes_pr_auc_guardrail"] = summary["pr_auc"].astype(float) >= float(tabular_pr - pr_guardrail_drop)
    summary["passes_topk_guardrail"] = summary["top_5pct_precision"].astype(float) >= tabular_top5
    summary["recommended"] = (
        summary["passes_roc_auc_threshold"]
        & summary["passes_pr_auc_guardrail"]
        & summary["passes_topk_guardrail"]
    )

    if not bool(summary["recommended"].any()):
        simplicity_order = {
            "tabular_only": 0,
            "torch_late_fusion": 1,
            "two_tower_cnn": 2,
            "two_tower_tcn": 3,
            "two_tower_cnn_gated": 4,
            "two_tower_tcn_gated": 5,
            "three_tower_cnn": 6,
            "three_tower_cnn_gated": 7,
            "three_tower_transformer_gated": 8,
            "three_tower_hybrid_gated": 9,
            "tabular_plus_p4205_cross_mlp": 10,
            "p4205_feature_only_mlp": 11,
            "torch_stacking_fusion": 12,
            "wave_only_cnn": 13,
            "wave_only_tcn": 14,
            "wave_only_transformer": 15,
        }
        summary["_simplicity_rank"] = summary["model_name"].map(simplicity_order).fillna(99)
        fallback_index = summary.sort_values(["roc_auc", "_simplicity_rank"], ascending=[False, True]).index[0]
        summary.loc[fallback_index, "recommended"] = True
        summary = summary.drop(columns=["_simplicity_rank"])

    summary = summary.sort_values(["roc_auc", "pr_auc", "top_5pct_precision"], ascending=False)
    summary.to_csv(base_output / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    (base_output / "metrics_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_model(
    model_name: str,
    tabular_dim: int,
    dropout: float,
    wave_in_channels: int,
    p4205_feature_dim: int | None = None,
) -> nn.Module:
    if model_name == "tabular_only":
        return BinaryTabularOnlyClassifier(tabular_in_dim=tabular_dim, dropout=dropout)
    if model_name == "tabular_plus_p4205_cross_mlp":
        return BinaryTabularOnlyClassifier(tabular_in_dim=tabular_dim, dropout=dropout)
    if model_name == "p4205_feature_only_mlp":
        return BinaryTabularOnlyClassifier(tabular_in_dim=tabular_dim, dropout=dropout)
    if model_name == "wave_only_cnn":
        return BinaryWaveOnlyClassifier(wave_encoder="cnn", dropout=dropout, wave_in_channels=wave_in_channels)
    if model_name == "wave_only_tcn":
        return BinaryWaveOnlyClassifier(wave_encoder="tcn", dropout=dropout, wave_in_channels=wave_in_channels)
    if model_name == "wave_only_transformer":
        return BinaryWaveOnlyClassifier(wave_encoder="transformer", dropout=dropout, wave_in_channels=wave_in_channels)
    if model_name == "two_tower_cnn":
        return BinaryTwoTowerClassifier(
            tabular_in_dim=tabular_dim,
            wave_encoder="cnn",
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
    if model_name == "two_tower_tcn":
        return BinaryTwoTowerClassifier(
            tabular_in_dim=tabular_dim,
            wave_encoder="tcn",
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
    if model_name == "two_tower_cnn_gated":
        return BinaryGatedTwoTowerClassifier(
            tabular_in_dim=tabular_dim,
            wave_encoder="cnn",
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
    if model_name == "two_tower_tcn_gated":
        return BinaryGatedTwoTowerClassifier(
            tabular_in_dim=tabular_dim,
            wave_encoder="tcn",
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
    if model_name == "three_tower_cnn":
        if p4205_feature_dim is None:
            raise ValueError("three_tower_cnn 需要 p4205_feature_dim。")
        return BinaryThreeTowerClassifier(
            tabular_in_dim=tabular_dim,
            p4205_feature_dim=p4205_feature_dim,
            wave_encoder="cnn",
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
    if model_name == "three_tower_cnn_gated":
        if p4205_feature_dim is None:
            raise ValueError("three_tower_cnn_gated 需要 p4205_feature_dim。")
        return BinaryGatedThreeTowerClassifier(
            tabular_in_dim=tabular_dim,
            p4205_feature_dim=p4205_feature_dim,
            wave_encoder="cnn",
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
    if model_name == "three_tower_transformer_gated":
        if p4205_feature_dim is None:
            raise ValueError("three_tower_transformer_gated 需要 p4205_feature_dim。")
        return BinaryGatedThreeTowerClassifier(
            tabular_in_dim=tabular_dim,
            p4205_feature_dim=p4205_feature_dim,
            wave_encoder="transformer",
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
    if model_name == "three_tower_hybrid_gated":
        if p4205_feature_dim is None:
            raise ValueError("three_tower_hybrid_gated 需要 p4205_feature_dim。")
        return BinaryGatedThreeTowerClassifier(
            tabular_in_dim=tabular_dim,
            p4205_feature_dim=p4205_feature_dim,
            wave_encoder="transformer",
            dropout=dropout,
            wave_in_channels=wave_in_channels,
        )
    raise ValueError(f"未知模型：{model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-Tower Torch 10年死亡风险二分类")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort", choices=COHORT_CHOICES, default="all_p4205_valid")
    parser.add_argument("--target-years", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-metric", choices=["pr_auc", "roc_auc", "balanced_accuracy"], default="roc_auc")
    parser.add_argument("--loss", choices=["focal_bce", "weighted_bce", "weighted_ce"], default="weighted_bce")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--roc-auc-baseline", type=float, default=0.7434)
    parser.add_argument("--min-roc-gain", type=float, default=0.003)
    parser.add_argument("--pr-guardrail-drop", type=float, default=0.002)
    parser.add_argument("--expected-length", type=int, default=100)
    parser.add_argument("--wave-channels", type=int, choices=[1, 2, 4, 5], default=2)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--max-rows", type=int, default=None, help="仅用于快速调试")
    parser.add_argument("--late-fusion-max-weight", type=float, default=0.40)
    parser.add_argument("--late-fusion-step", type=float, default=0.02)
    parser.add_argument("--disable-late-fusion", action="store_true")
    parser.add_argument("--disable-stacking-fusion", action="store_true")
    parser.add_argument("--oof-splits", type=int, default=5)
    parser.add_argument("--p4205-cache", type=Path, default=DEFAULT_P4205_CACHE)
    parser.add_argument("--disable-p4205-cache", action="store_true")
    parser.add_argument("--disable-p4205-nonlinear", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_CHOICES,
        choices=MODEL_CHOICES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p4205_cache_path = None if args.disable_p4205_cache else args.p4205_cache

    df = load_data(args.data, args.target_years, args.max_rows)
    df = apply_cohort_filter(df, args.cohort, p4205_cache_path)
    y = df["target"].to_numpy(dtype=np.int64)
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=args.val_size, stratify=y[train_val_idx], random_state=RANDOM_STATE
    )

    structured = build_structured_frame(df)
    preprocessor, numeric_columns, categorical_columns = build_tabular_preprocessor(structured.iloc[train_idx])
    x_train_tab = preprocessor.fit_transform(structured.iloc[train_idx]).astype(np.float32, copy=False)
    x_val_tab = preprocessor.transform(structured.iloc[val_idx]).astype(np.float32, copy=False)
    x_test_tab = preprocessor.transform(structured.iloc[test_idx]).astype(np.float32, copy=False)

    p4205_feature_frame = build_p4205_feature_tower_frame(
        df[WAVE_COLUMN],
        eid_values=df[EID_COLUMN],
        cache_path=p4205_cache_path,
        include_nonlinear=not args.disable_p4205_nonlinear,
    )
    p4205_preprocessor = build_numeric_preprocessor(p4205_feature_frame.iloc[train_idx])
    x_train_p4205 = p4205_preprocessor.fit_transform(p4205_feature_frame.iloc[train_idx]).astype(np.float32, copy=False)
    x_val_p4205 = p4205_preprocessor.transform(p4205_feature_frame.iloc[val_idx]).astype(np.float32, copy=False)
    x_test_p4205 = p4205_preprocessor.transform(p4205_feature_frame.iloc[test_idx]).astype(np.float32, copy=False)

    y_train_for_oof = y[train_idx]
    p4205_oof_train, p4205_risk_val, p4205_risk_test = make_oof_p4205_risk(
        x_train_p4205,
        y_train_for_oof,
        x_val_p4205,
        x_test_p4205,
        args.oof_splits,
    )
    cross_train_frame = build_cross_feature_frame(structured.iloc[train_idx], p4205_oof_train)
    cross_val_frame = build_cross_feature_frame(structured.iloc[val_idx], p4205_risk_val)
    cross_test_frame = build_cross_feature_frame(structured.iloc[test_idx], p4205_risk_test)
    cross_preprocessor, cross_numeric_columns, cross_categorical_columns = build_tabular_preprocessor(cross_train_frame)
    x_train_cross = cross_preprocessor.fit_transform(cross_train_frame).astype(np.float32, copy=False)
    x_val_cross = cross_preprocessor.transform(cross_val_frame).astype(np.float32, copy=False)
    x_test_cross = cross_preprocessor.transform(cross_test_frame).astype(np.float32, copy=False)

    uses_transformer_wave = any("transformer" in name or name == "three_tower_hybrid_gated" for name in args.models)
    effective_wave_channels = 5 if uses_transformer_wave else int(args.wave_channels)
    waveform_config = WaveformConfig(
        expected_length=args.expected_length,
        normalize=not args.no_normalize,
        return_channels=effective_wave_channels > 1,
        channel_count=effective_wave_channels,
    )

    x_train_wave = df[WAVE_COLUMN].iloc[train_idx].tolist()
    x_val_wave = df[WAVE_COLUMN].iloc[val_idx].tolist()
    x_test_wave = df[WAVE_COLUMN].iloc[test_idx].tolist()
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    eid_train = df[EID_COLUMN].iloc[train_idx].to_numpy()
    eid_val = df[EID_COLUMN].iloc[val_idx].to_numpy()
    eid_test = df[EID_COLUMN].iloc[test_idx].to_numpy()

    preprocess_summary = {
        "target_definition": f"基线后 {args.target_years:g} 年内全因死亡",
        "target_caveat": "若阴性参与者不足目标年限随访，应改用生存分析。",
        "row_count": int(len(df)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "cohort": args.cohort,
        "structured_columns": BASELINE_COLUMNS,
        "engineered_feature_count_before_encode": int(structured.shape[1]),
        "tabular_feature_dim_after_encode": int(x_train_tab.shape[1]),
        "p4205_feature_count_before_encode": int(p4205_feature_frame.shape[1]),
        "p4205_feature_dim_after_encode": int(x_train_p4205.shape[1]),
        "cross_feature_dim_after_encode": int(x_train_cross.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "cross_numeric_columns": cross_numeric_columns,
        "cross_categorical_columns": cross_categorical_columns,
        "p4205_feature_sources": {
            "existing_summary_count": len(WAVE_FEATURE_COLUMNS),
            "advanced_feature_count": len(ADVANCED_FEATURE_COLUMNS),
            "raw_diff_stat_count": int(build_raw_diff_stat_frame(df[WAVE_COLUMN].head(1)).shape[1]),
            "advanced_cache_path": str(p4205_cache_path) if p4205_cache_path is not None else None,
            "advanced_cache_used": bool(
                p4205_cache_path is not None
                and Path(p4205_cache_path).exists()
                and EID_COLUMN in pd.read_csv(p4205_cache_path, nrows=0).columns
            ),
            "include_nonlinear_when_recompute": bool(not args.disable_p4205_nonlinear),
        },
        "wave_channels": int(effective_wave_channels),
        "requested_wave_channels": int(args.wave_channels),
        "uses_transformer_wave": bool(uses_transformer_wave),
        "device": str(device),
        "models": args.models,
        "loss": args.loss,
        "early_stop_metric": args.early_stop_metric,
        "min_delta": args.min_delta,
        "roc_auc_baseline": args.roc_auc_baseline,
        "min_roc_gain": args.min_roc_gain,
        "pr_guardrail_drop": args.pr_guardrail_drop,
        "oof_splits": args.oof_splits,
    }
    (output_dir / "preprocess_summary.json").write_text(
        json.dumps(preprocess_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_two = TwoTowerBinaryDataset(eid_train, x_train_wave, x_train_tab, y_train, waveform_config)
    val_two = TwoTowerBinaryDataset(eid_val, x_val_wave, x_val_tab, y_val, waveform_config)
    test_two = TwoTowerBinaryDataset(eid_test, x_test_wave, x_test_tab, y_test, waveform_config)
    train_three = ThreeTowerBinaryDataset(
        eid_train, x_train_wave, x_train_p4205, x_train_tab, y_train, waveform_config
    )
    val_three = ThreeTowerBinaryDataset(eid_val, x_val_wave, x_val_p4205, x_val_tab, y_val, waveform_config)
    test_three = ThreeTowerBinaryDataset(eid_test, x_test_wave, x_test_p4205, x_test_tab, y_test, waveform_config)
    train_wave = WaveOnlyBinaryDataset(eid_train, x_train_wave, y_train, waveform_config)
    val_wave = WaveOnlyBinaryDataset(eid_val, x_val_wave, y_val, waveform_config)
    test_wave = WaveOnlyBinaryDataset(eid_test, x_test_wave, y_test, waveform_config)
    train_tab = TabularOnlyBinaryDataset(eid_train, x_train_tab, y_train)
    val_tab = TabularOnlyBinaryDataset(eid_val, x_val_tab, y_val)
    test_tab = TabularOnlyBinaryDataset(eid_test, x_test_tab, y_test)
    train_cross = TabularOnlyBinaryDataset(eid_train, x_train_cross, y_train)
    val_cross = TabularOnlyBinaryDataset(eid_val, x_val_cross, y_val)
    test_cross = TabularOnlyBinaryDataset(eid_test, x_test_cross, y_test)
    train_p4205_feature = TabularOnlyBinaryDataset(eid_train, x_train_p4205, y_train)
    val_p4205_feature = TabularOnlyBinaryDataset(eid_val, x_val_p4205, y_val)
    test_p4205_feature = TabularOnlyBinaryDataset(eid_test, x_test_p4205, y_test)

    loaders = {
        "two": (
            make_loader(train_two, args.batch_size, True),
            make_loader(val_two, args.batch_size, False),
            make_loader(test_two, args.batch_size, False),
        ),
        "three": (
            make_loader(train_three, args.batch_size, True),
            make_loader(val_three, args.batch_size, False),
            make_loader(test_three, args.batch_size, False),
        ),
        "wave": (
            make_loader(train_wave, args.batch_size, True),
            make_loader(val_wave, args.batch_size, False),
            make_loader(test_wave, args.batch_size, False),
        ),
        "tab": (
            make_loader(train_tab, args.batch_size, True),
            make_loader(val_tab, args.batch_size, False),
            make_loader(test_tab, args.batch_size, False),
        ),
        "cross": (
            make_loader(train_cross, args.batch_size, True),
            make_loader(val_cross, args.batch_size, False),
            make_loader(test_cross, args.batch_size, False),
        ),
        "p4205_feature": (
            make_loader(train_p4205_feature, args.batch_size, True),
            make_loader(val_p4205_feature, args.batch_size, False),
            make_loader(test_p4205_feature, args.batch_size, False),
        ),
    }

    model_payloads: dict[str, dict[str, object]] = {}
    for model_name in args.models:
        if model_name == "tabular_plus_p4205_cross_mlp":
            current_tab_dim = int(x_train_cross.shape[1])
        elif model_name == "p4205_feature_only_mlp":
            current_tab_dim = int(x_train_p4205.shape[1])
        else:
            current_tab_dim = int(x_train_tab.shape[1])
        model = build_model(
            model_name,
            current_tab_dim,
            args.dropout,
            effective_wave_channels,
            p4205_feature_dim=int(x_train_p4205.shape[1]),
        ).to(device)
        if model_name.startswith("three_tower"):
            selected_loaders = loaders["three"]
        elif model_name.startswith("two_tower"):
            selected_loaders = loaders["two"]
        elif model_name.startswith("wave_only"):
            selected_loaders = loaders["wave"]
        elif model_name == "tabular_plus_p4205_cross_mlp":
            selected_loaders = loaders["cross"]
        elif model_name == "p4205_feature_only_mlp":
            selected_loaders = loaders["p4205_feature"]
        else:
            selected_loaders = loaders["tab"]

        train_loader, val_loader, test_loader = selected_loaders
        payload = train_one_model(
            model=model,
            model_name=model_name,
            output_dir=output_dir / model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            val_eids=eid_val,
            test_eids=eid_test,
            train_labels=y_train,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            patience=args.patience,
            device=device,
            model_config={
                "name": model_name,
                "target_years": args.target_years,
                "dropout": args.dropout,
                "tabular_in_dim": current_tab_dim if not model_name.startswith("wave_only") else None,
                "p4205_feature_dim": int(x_train_p4205.shape[1]) if model_name.startswith("three_tower") else None,
                "wave_expected_length": args.expected_length if model_name != "tabular_only" else None,
                "wave_channels": effective_wave_channels if model_name != "tabular_only" else None,
                "uses_oof_p4205_cross_features": model_name == "tabular_plus_p4205_cross_mlp",
            },
            loss_name=args.loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            early_stop_metric=args.early_stop_metric,
            min_delta=args.min_delta,
        )
        model_payloads[model_name] = payload

    if not args.disable_late_fusion and {"tabular_only", "wave_only_cnn"}.issubset(model_payloads):
        model_payloads[LATE_FUSION_NAME] = export_late_fusion(
            output_dir,
            model_payloads["tabular_only"],
            model_payloads["wave_only_cnn"],
            eid_val,
            eid_test,
            args.late_fusion_max_weight,
            args.late_fusion_step,
        )

    if not args.disable_stacking_fusion:
        stacking_payload = export_stacking_fusion(
            output_dir,
            model_payloads,
            structured,
            val_idx,
            test_idx,
            eid_val,
            eid_test,
        )
        if stacking_payload is not None:
            model_payloads[STACKING_FUSION_NAME] = stacking_payload

    aggregate_metrics(
        output_dir,
        model_payloads,
        roc_auc_baseline=args.roc_auc_baseline,
        min_roc_gain=args.min_roc_gain,
        pr_guardrail_drop=args.pr_guardrail_drop,
    )
    public_payloads = {name: public_metric_payload(payload) for name, payload in model_payloads.items()}
    (output_dir / "run_results.json").write_text(
        json.dumps(public_payloads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Two-Tower 10年死亡风险训练完成：{output_dir}")


if __name__ == "__main__":
    main()

