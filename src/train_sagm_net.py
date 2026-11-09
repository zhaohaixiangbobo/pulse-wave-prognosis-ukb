"""
文件说明：训练和评估 SAGM-Net，用于 P4205 十年死亡风险深度学习主架构实验。

本脚本固定 P4205 有效队列的 train/validation/test 切分，支持严格消融、SAGM 门控主模型、
多任务死因辅助学习、gate 可解释性分析、bootstrap 区间和论文主表导出。
"""

from __future__ import annotations
from two_tower_torch_model import RANDOM_STATE, WaveformConfig, parse_p4205_i0_to_tensor, set_global_seed
from train_two_tower_classification import (
    AGE_COLUMN,
    DEFAULT_DATA,
    DEFAULT_P4205_CACHE,
    DEATH_AGE_COLUMN,
    EID_COLUMN,
    WAVE_COLUMN,
    FocalBCEWithLogitsLoss,
    apply_cohort_filter,
    build_numeric_preprocessor,
    build_p4205_feature_tower_frame,
    build_structured_frame,
    build_tabular_preprocessor,
    load_data,
    topk_metrics,
)
from sagm_followup_analysis import age_split_distribution, grouped_permutation_importance, summarize_permutation_importance
from landmark_followup import attach_landmark_followup, load_followup_workbook
from sagm_torch_model import SAGMNet
from sagm_cvd_endpoints import (
    CAUSE_COLUMN,
    DIAGNOSIS_COLUMN,
    TASK_LABELS,
    add_primary_cause_columns,
    build_labels as build_endpoint_labels,
    hierarchy_strata,
    task_label_map,
    task_names,
)
from p4205_feature_policy import (
    STRUCTURED_FEATURE_SET_CHOICES,
    is_age_derived_structured_feature,
    is_p4205_derived_structured_feature,
    split_structured_features,
)

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "models") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "models"))


DEFAULT_OUTPUT = PROJECT_ROOT / "data/model_output_sagm_main_seed42"
ABLATION_MODES = (
    "structured_only",
    "raw_waveform_only",
    "engineered_p4205_only",
    "engineered_p4205_cross_only",
    "two_tower_raw",
    "structured_engineered",
    "structured_p4205_cross",
    "three_tower_concat",
    "sagm_gated",
    "sagm_vector_gated",
    "sagm_gated_raw_only",
    "sagm_gated_feature_only",
    "sagm_gated_cross_feature",
    "sagm_gated_cross_three_tower",
    "sagm_residual_delta",
)
WAVE_ENCODER_DEPENDENT_MODES = {
    "raw_waveform_only",
    "two_tower_raw",
    "three_tower_concat",
    "sagm_gated",
    "sagm_vector_gated",
    "sagm_gated_raw_only",
    "sagm_gated_cross_three_tower",
    "sagm_residual_delta",
}
P4205_CROSS_MODE_CHOICES = ("none", "selected_clinical_x_p4205")
P4205_CROSS_ABLATION_MODES = {
    "engineered_p4205_cross_only",
    "structured_p4205_cross",
    "sagm_gated_cross_feature",
    "sagm_gated_cross_three_tower",
}
SELECTED_CLINICAL_CROSS_COLUMNS = (
    "p21022", "age_squared", "p20162_i0", "symptom_yes_count", "sex_male")
SELECTED_P4205_CROSS_COLUMNS = (
    "p4205_stat_raw_std",
    "p4205_stat_raw_iqr",
    "p4205_stat_raw_cv",
    "p4205_time_h3_h1_ratio",
    "p4205_time_h4_h1_ratio",
    "p4205_time_h5_h1_ratio",
    "p4205_time_systolic_diastolic_area_ratio",
    "p4205_morph_max_rise",
    "p4205_morph_max_fall",
    "p4205_morph_shoulder_strength",
    "p4205_morph_symmetry_ratio",
    "p4205_freq_spectral_entropy",
    "p4205_freq_low_high_ratio",
    "p4205_nonlinear_sample_entropy",
)


class SAGMDataset(Dataset):
    """SAGM 数据集：波形、P4205 engineered features、结构化特征和多任务标签。"""

    def __init__(
        self,
        eid_values: np.ndarray,
        wave_values: list[object],
        p4205_features: np.ndarray,
        p4205_cross_features: np.ndarray,
        structured_features: np.ndarray,
        labels: np.ndarray,
        label_mask: np.ndarray,
        waveform_config: WaveformConfig,
    ) -> None:
        if not (
            len(wave_values)
            == p4205_features.shape[0]
            == p4205_cross_features.shape[0]
            == structured_features.shape[0]
            == labels.shape[0]
            == label_mask.shape[0]
        ):
            raise ValueError("SAGM 输入样本数不一致。")
        self.eid_values = np.asarray(eid_values)
        self.wave_values = wave_values
        self.p4205_features = p4205_features.astype(np.float32, copy=False)
        self.p4205_cross_features = p4205_cross_features.astype(
            np.float32, copy=False)
        self.structured_features = structured_features.astype(
            np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.label_mask = label_mask.astype(np.float32, copy=False)
        self.waveform_config = waveform_config
        self.cached_wave = None
        self.wave_enabled = False
        self.empty_wave = torch.empty(0, dtype=torch.float32)

    def configure_wave(self, enabled: bool) -> None:
        """仅波形模型构建一次缓存；按样本原函数解析，训练和诊断复用。"""
        self.wave_enabled = enabled
        if enabled and self.cached_wave is None:
            started = time.time()
            count = len(self.wave_values)
            shape = parse_p4205_i0_to_tensor(
                self.wave_values[0], config=self.waveform_config).shape
            self.cached_wave = torch.empty(
                (count, *shape), dtype=torch.float32)
            for i, value in enumerate(self.wave_values):
                self.cached_wave[i].copy_(parse_p4205_i0_to_tensor(
                    value, config=self.waveform_config))
                if (i + 1) % 10000 == 0 or i + 1 == count:
                    print(
                        f'[wave cache] {i+1}/{count} {(i+1)/count:.1%} elapsed={time.time()-started:.1f}s', flush=True)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wave = self.cached_wave[idx] if self.wave_enabled else self.empty_wave
        p4205 = torch.as_tensor(self.p4205_features[idx], dtype=torch.float32)
        p4205_cross = torch.as_tensor(
            self.p4205_cross_features[idx], dtype=torch.float32)
        structured = torch.as_tensor(
            self.structured_features[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        mask = torch.as_tensor(self.label_mask[idx], dtype=torch.float32)
        return wave, p4205, p4205_cross, structured, label, mask


class MaskedWeightedBCEWithLogitsLoss(nn.Module):
    """支持多任务缺失标签 mask 的加权 BCE。"""

    def __init__(self, pos_weight: torch.Tensor, task_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("task_weights", task_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # masked 任务不参与 loss，但 BCE 会先读取 target；这里先清理历史 NaN 标签。
        clean_target = torch.where(mask > 0, torch.nan_to_num(
            target, nan=0.0), torch.zeros_like(target))
        raw = F.binary_cross_entropy_with_logits(
            logits, clean_target, pos_weight=self.pos_weight, reduction="none")
        weighted = raw * mask * self.task_weights
        denominator = (mask * self.task_weights).sum().clamp_min(1.0)
        return weighted.sum() / denominator


class MaskedFocalBCEWithLogitsLoss(nn.Module):
    """支持多任务缺失标签 mask 的 Focal BCE。"""

    def __init__(self, alpha: float, gamma: float, pos_weight: torch.Tensor, task_weights: torch.Tensor) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("task_weights", task_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # masked 任务不参与 loss，但 BCE 会先读取 target；这里先清理历史 NaN 标签。
        clean_target = torch.where(mask > 0, torch.nan_to_num(
            target, nan=0.0), torch.zeros_like(target))
        bce = F.binary_cross_entropy_with_logits(
            logits, clean_target, pos_weight=self.pos_weight, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = probability * clean_target + \
            (1.0 - probability) * (1.0 - clean_target)
        alpha_t = self.alpha * clean_target + \
            (1.0 - self.alpha) * (1.0 - clean_target)
        raw = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        weighted = raw * mask * self.task_weights
        denominator = (mask * self.task_weights).sum().clamp_min(1.0)
        return weighted.sum() / denominator


def parse_inception_kernel_sets(value: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """解析三阶段 InceptionTime-lite 卷积核，格式为 ``3,7,15;3,9,21;3,11,31``。"""
    try:
        stages = tuple(tuple(int(item.strip()) for item in stage.split(","))
                       for stage in value.split(";"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Inception 卷积核必须是以逗号和分号分隔的整数。") from error
    if len(stages) != 3 or any(len(stage) != 3 for stage in stages):
        raise argparse.ArgumentTypeError("Inception 卷积核必须为 3 个阶段，每阶段 3 个数。")
    if any(kernel <= 0 or kernel % 2 == 0 for stage in stages for kernel in stage):
        raise argparse.ArgumentTypeError("Inception 卷积核必须是正奇数。")
    return stages  # type: ignore[return-value]


def parse_inception_stage_channels(value: str) -> tuple[int, int, int]:
    """解析三阶段通道数，格式为 ``64,96,128``。"""
    try:
        channels = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Inception 阶段通道数必须是以逗号分隔的整数。") from error
    if len(channels) != 3 or any(channel <= 0 for channel in channels):
        raise argparse.ArgumentTypeError("Inception 阶段通道数必须为 3 个正整数。")
    return channels  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 SAGM-Net P4205 深度学习主架构")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--p4205-cache", type=Path,
                        default=DEFAULT_P4205_CACHE)
    parser.add_argument("--cohort", default="all_p4205_valid")
    parser.add_argument("--target-years", type=float, default=10.0)
    parser.add_argument("--task-mode", choices=["binary", "mtl", "cvd_death_mtl",
                        "cvd_death_resp_mtl", "cvd_diagnosis_mtl", "landmark_death_mtl"], default="binary")
    parser.add_argument("--followup-dates", type=Path, default=None,
                        help="landmark_death_mtl 必需的 followup_dates.xlsx")
    parser.add_argument(
        "--structured-feature-set",
        choices=list(STRUCTURED_FEATURE_SET_CHOICES),
        default="current",
        help="结构化基线口径；clinical_no_p4205 会剔除 P4205 派生结构化摘要。",
    )
    parser.add_argument(
        "--structured-encoder",
        choices=["mlp", "vsn"],
        default="mlp",
        help="结构化塔编码器；默认 mlp 保持历史实验可复现。",
    )
    parser.add_argument(
        "--p4205-feature-encoder",
        choices=["mlp", "vsn"],
        default="mlp",
        help="P4205 工程特征塔编码器；默认 mlp 保持历史实验可复现。",
    )
    parser.add_argument("--vsn-variable-dim", type=int,
                        default=32, help="VSN 单变量表征维度。")
    parser.add_argument("--models", nargs="+",
                        choices=ABLATION_MODES, default=list(ABLATION_MODES))
    parser.add_argument("--wave-encoders", nargs="+", choices=[
                        "cnn", "resnet1d", "patchtst", "inception_lite"], default=["resnet1d"])
    parser.add_argument(
        "--inception-kernel-sets",
        type=parse_inception_kernel_sets,
        default="3,7,15;3,9,21;3,11,31",
        help="三阶段 Inception 卷积核：例如 3,7,15;3,9,21;3,11,31。仅 inception_lite 生效。",
    )
    parser.add_argument(
        "--inception-stage-channels",
        type=parse_inception_stage_channels,
        default="64,96,128",
        help="三阶段 Inception 通道数：例如 64,96,128。仅 inception_lite 生效。",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument(
        "--loss", choices=["weighted_bce", "focal_bce"], default="weighted_bce")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--task-weights",
        type=str,
        default=None,
        help="可选逗号分隔任务权重；留空时按 task-mode 使用预设科研口径。",
    )
    parser.add_argument(
        "--early-stop-aggregation",
        choices=["auto", "all_cause", "macro_tasks", "landmark_all_cause"],
        default="auto",
        help="诊断多任务默认对各头 PR/ROC 分数取宏平均；其他模式保持全因死亡早停。",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument(
        "--early-stop-metric",
        choices=["pr_auc", "roc_auc", "mean_pr_roc"],
        default="pr_auc",
        help="早停指标：默认保持 PR-AUC，审计复验可切换 ROC-AUC 或 PR/ROC 均值。",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="训练随机种子；默认沿用 --seed。指定后只影响初始化和 DataLoader，不改变冻结切分。",
    )
    parser.add_argument(
        "--split-assignments",
        type=Path,
        default=None,
        help="可选的冻结 split_assignments.csv；提供后严格按 eid 复用 train/validation/test 切分。",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--expected-length", type=int, default=100)
    parser.add_argument("--wave-channels", type=int,
                        choices=[1, 2, 4, 5], default=4)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--disable-bootstrap", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="保存每个模型的最佳 state_dict 及配置快照。")
    parser.add_argument("--permutation-repeats", type=int,
                        default=0, help="SAGM 测试时分组置换归因重复次数；0 表示关闭。")
    parser.add_argument("--permutation-max-samples", type=int,
                        default=0, help="置换归因样本上限；0 表示使用完整测试集。")
    parser.add_argument(
        "--p4205-cross-mode",
        choices=P4205_CROSS_MODE_CHOICES,
        default="none",
        help="P4205 分支早期交叉特征模式；交叉特征只进入 P4205/cross 分支，不进入 clinical-only baseline。",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_sagm_output_path(path_value: str | Path) -> Path:
    """解析 SAGM 输出目录，允许用户传入 data/model_output_x 或 model_output_x。"""
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    parts = raw.parts
    if parts and parts[0].lower() == "data":
        return (PROJECT_ROOT / raw).resolve()
    return (PROJECT_ROOT / "data" / raw).resolve()


def add_endpoint_columns(frame: pd.DataFrame, data_path: Path, max_rows: int | None, task_mode: str) -> pd.DataFrame:
    """按任务模式回连结局字段；这些字段只用于标签，后续绝不传给特征构建器。"""
    usecols = [EID_COLUMN, CAUSE_COLUMN]
    if task_mode == "cvd_diagnosis_mtl":
        usecols.append(DIAGNOSIS_COLUMN)
    endpoint = pd.read_csv(data_path, usecols=usecols,
                           nrows=max_rows, low_memory=False)
    if endpoint[EID_COLUMN].duplicated().any():
        raise ValueError("原始数据 eid 不唯一，无法安全回连结局字段。")
    merged = frame.merge(endpoint, on=EID_COLUMN,
                         how="left", validate="one_to_one")
    return add_primary_cause_columns(merged)


def build_multitask_labels(frame: pd.DataFrame, task_mode: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """委托可单测的 ICD 模块构建标签和 mask，保持旧 binary/mtl 接口兼容。"""
    return build_endpoint_labels(frame, task_mode)


def resolve_task_weights(task_mode: str, value: str | None, task_count: int) -> np.ndarray:
    """解析任务权重；预设值保证新增嵌套心血管头不会淹没全因死亡主任务。"""
    defaults = {
        "binary": (1.0,),
        "mtl": (1.0, 0.7, 0.5, 0.5),
        "cvd_death_mtl": (1.0, 0.7, 0.35, 0.35, 0.5, 0.5),
        "cvd_death_resp_mtl": (1.0, 0.7, 0.35, 0.35, 0.25, 0.15, 0.15, 0.15, 0.5),
        "cvd_diagnosis_mtl": (1.0, 1.0, 1.0),
        "landmark_death_mtl": (1.0, 0.35, 1.0, 0.35),
    }
    weights = tuple(float(item.strip())
                    for item in value.split(",")) if value else defaults[task_mode]
    if len(weights) != task_count or any(weight <= 0 for weight in weights):
        raise ValueError(f"task 权重必须包含 {task_count} 个正数，实际为 {weights}。")
    return np.asarray(weights, dtype=np.float32)


def resolve_early_stop_aggregation(task_mode: str, value: str) -> str:
    """诊断头没有全因主任务，自动使用多头宏平均；历史模式保持原有 all_cause 行为。"""
    if value != "auto":
        return value
    if task_mode == "cvd_diagnosis_mtl":
        return "macro_tasks"
    if task_mode == "landmark_death_mtl":
        return "landmark_all_cause"
    return "all_cause"


def primary_task_name(tasks: list[str], aggregation: str) -> str:
    """确定主展示任务；宏平均诊断实验以宽口径头作为单头展示参考。"""
    if aggregation == "all_cause" and "all_cause" in tasks:
        return "all_cause"
    if aggregation == "landmark_all_cause" and "all_cause_0_5" in tasks:
        return "all_cause_0_5"
    return "cv_wide_I00_I99" if "cv_wide_I00_I99" in tasks else tasks[0]


def safe_binary_int_array(values: np.ndarray) -> np.ndarray:
    """将可能包含 NaN 的二值数组安全转为 int8，用于 CSV 导出。"""
    cleaned = np.nan_to_num(np.asarray(
        values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned.astype(np.int8)


def assert_disjoint_split_eids(frame: pd.DataFrame, split_indices: dict[str, np.ndarray]) -> None:
    """断言 train/validation/test 的 eid 没有交叉，避免同一受试者跨 split 泄漏。"""
    eid_sets: dict[str, set[object]] = {}
    for split, idx in split_indices.items():
        eid_sets[split] = set(frame[EID_COLUMN].iloc[idx].dropna().tolist())
    split_names = list(eid_sets)
    for left_pos, left in enumerate(split_names):
        for right in split_names[left_pos + 1:]:
            overlap = eid_sets[left].intersection(eid_sets[right])
            if overlap:
                sample = list(overlap)[:10]
                raise ValueError(
                    f"eid overlap between {left} and {right}: n={len(overlap)}, sample={sample}")


def load_frozen_split_indices(frame: pd.DataFrame, assignment_path: Path) -> dict[str, np.ndarray]:
    """按 eid 读取冻结切分，拒绝缺失、重复和跨 split 泄漏。"""
    if not assignment_path.is_file():
        raise FileNotFoundError(f"未找到冻结切分文件：{assignment_path}")
    assignment = pd.read_csv(assignment_path, usecols=[EID_COLUMN, "split"], dtype={
                             EID_COLUMN: "string", "split": "string"})
    expected = {"train", "validation", "test"}
    actual = set(assignment["split"].dropna().astype(str))
    if actual != expected:
        raise ValueError(
            f"冻结切分必须恰好包含 {sorted(expected)}，实际为 {sorted(actual)}。")
    if assignment[EID_COLUMN].isna().any() or assignment.duplicated(EID_COLUMN).any():
        raise ValueError("冻结切分存在空 eid 或重复 eid。")
    frame_eids = frame[EID_COLUMN].astype("string")
    if frame_eids.isna().any() or frame_eids.duplicated().any():
        raise ValueError("当前队列存在空 eid 或重复 eid，无法安全复用冻结切分。")
    if set(assignment[EID_COLUMN].astype(str)) != set(frame_eids.astype(str)):
        raise ValueError("冻结切分 eid 集合与当前队列不一致，已停止以避免错配。")
    lookup = pd.Series(np.arange(len(frame), dtype=np.int64),
                       index=frame_eids.astype(str))
    split_indices: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        split_eids = assignment.loc[assignment["split"].eq(
            split), EID_COLUMN].astype(str)
        split_indices[split] = lookup.loc[split_eids].to_numpy(dtype=np.int64)
    assert_disjoint_split_eids(frame, split_indices)
    return split_indices


def build_generated_split_indices(
    labels: np.ndarray,
    masks: np.ndarray,
    tasks: list[str],
    task_mode: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """按嵌套心血管结局的互斥层级切分，保证 train/validation/test 中稀有结局均衡。"""
    strata = hierarchy_strata(tasks, labels, masks, task_mode)
    values, counts = np.unique(strata, return_counts=True)
    if counts.min() < 3:
        detail = {int(value): int(count)
                  for value, count in zip(values, counts, strict=True)}
        raise ValueError(f"分层类别样本过少，无法安全三段切分：{detail}")
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=strata, random_state=seed)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=strata[train_val_idx],
        random_state=seed,
    )
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def save_split_diagnostics(
    output_dir: Path,
    frame: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    tasks: list[str],
    labels: np.ndarray,
    masks: np.ndarray,
) -> pd.DataFrame:
    """导出 split 明细和汇总，固定记录样本数、阳性率、mask 与 eid 唯一性。"""
    assert_disjoint_split_eids(frame, split_indices)
    assignment_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for split, idx in split_indices.items():
        split_eids = frame[EID_COLUMN].iloc[idx].to_numpy()
        part = pd.DataFrame(
            {EID_COLUMN: split_eids, "split": split, "row_index": idx})
        row: dict[str, object] = {
            "split": split,
            "n": int(len(idx)),
            "unique_eid": int(pd.Series(split_eids).nunique(dropna=True)),
            "duplicate_eid_count": int(len(idx) - pd.Series(split_eids).nunique(dropna=True)),
            "eid_unique": bool(pd.Series(split_eids).is_unique),
        }
        for task_index, task in enumerate(tasks):
            task_label = safe_binary_int_array(labels[idx, task_index])
            task_mask = safe_binary_int_array(masks[idx, task_index])
            keep = task_mask > 0
            positives = int(task_label[keep].sum()) if keep.any() else 0
            part[f"{task}_label"] = task_label
            part[f"{task}_mask"] = task_mask
            row[f"{task}_mask_count"] = int(keep.sum())
            row[f"{task}_positive_count"] = positives
            row[f"{task}_positive_rate"] = float(
                positives / max(1, int(keep.sum())))
        assignment_parts.append(part)
        summary_rows.append(row)
    assignment = pd.concat(assignment_parts, axis=0, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    assignment.to_csv(output_dir / "split_assignments.csv",
                      index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "split_summary.csv",
                   index=False, encoding="utf-8-sig")
    return summary


def write_model_audit_note(
    output_dir: Path,
    args: argparse.Namespace,
    config: dict[str, object],
    split_summary: pd.DataFrame,
) -> None:
    """写出本轮实现审计说明，便于论文和后续复验追踪口径。"""
    lines = [
        "# SAGM-Net Implementation Audit",
        "",
        "## Scope",
        "- 本审计说明由 `models/train_sagm_net.py` 自动生成，用于记录本轮早停、阈值评估和 split 诊断口径。",
        "- 当前保持 SAGM-Net 主结构不变：结构化特征为主干，P4205 raw waveform 和 engineered P4205 features 作为增量分支。",
        "",
        "## Split Checks",
        "- 脚本会断言 train/validation/test 的 `eid` 无交叉，并导出 `split_assignments.csv` 与 `split_summary.csv`。",
    ]
    for _, row in split_summary.iterrows():
        lines.append(
            f"- {row['split']}: n={int(row['n'])}, unique_eid={int(row['unique_eid'])}, "
            f"all_cause_positive={int(row.get('all_cause_positive_count', 0))}, "
            f"positive_rate={float(row.get('all_cause_positive_rate', 0.0)):.6f}"
        )
    lines.extend(
        [
            "",
            "## Early Stopping",
            f"- `early_stop_metric={args.early_stop_metric}`，可选 `pr_auc`、`roc_auc`、`mean_pr_roc`。",
            f"- `patience={args.patience}`，`min_delta={args.min_delta}`；训练历史会同时记录当前和最佳 PR-AUC/ROC-AUC。",
            "",
            "## Threshold Policy",
            "- validation split 选择阈值；test split 固定使用 validation 阈值。",
            "- ROC-AUC、PR-AUC、TopK precision/recall 不依赖阈值，是主比较指标。Balanced Accuracy、Sensitivity、Specificity 仅作为补充。",
            "",
            "## Model Audit",
            "- 当前 `SAGMNet` 按 ablation mode 只实例化需要的分支，避免无用分支参与参数和解释。",
            "- `sagm_gated` 使用 scalar gate，解释性直接，但表达能力可能弱于 concat。",
            "- `sagm_residual_delta` 会把结构化 logit 作为锚点，只让 P4205 分支学习残差修正，专门服务增量检验。",
            "- 若使用 `clinical_no_p4205`，结构化基线会剔除 p21021/p4199/p4200 及其派生项，便于评估 raw P4205 新增价值。",
            "",
            "## Run Config Snapshot",
            f"- task_mode={config['task_mode']}, cohort={config['cohort']}, target_years={config['target_years']}, seed={config['seed']}, device={config['device']}",
        ]
    )
    (output_dir / "SAGM_MODEL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """按列安全取数值；缺列时返回 NaN，后续由数值预处理器统一填补。"""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=np.float32)
    return pd.to_numeric(frame[column], errors="coerce").astype(np.float32)


def build_selected_p4205_cross_frame(structured: pd.DataFrame, p4205_frame: pd.DataFrame) -> pd.DataFrame:
    """构造小规模 clinical 与 P4205 乘积交叉，并保留在 P4205 侧特征表。"""
    result = p4205_frame.copy()
    clinical_values: dict[str, pd.Series] = {
        "p21022": _numeric_column(structured, "p21022"),
        "age_squared": _numeric_column(structured, "age_squared"),
        "p20162_i0": _numeric_column(structured, "p20162_i0"),
        "symptom_yes_count": _numeric_column(structured, "symptom_yes_count"),
    }
    sex = structured.get("sex", pd.Series(
        "", index=structured.index)).astype("string").str.lower()
    clinical_values["sex_male"] = sex.eq("male").astype(np.float32)
    for clinical_name in SELECTED_CLINICAL_CROSS_COLUMNS:
        clinical = clinical_values.get(clinical_name, pd.Series(
            np.nan, index=structured.index, dtype=np.float32))
        for p4205_name in SELECTED_P4205_CROSS_COLUMNS:
            p4205 = _numeric_column(p4205_frame, p4205_name)
            result[f"cross_{clinical_name}__{p4205_name}"] = clinical.to_numpy(
                dtype=np.float32) * p4205.to_numpy(dtype=np.float32)
    return result.replace([np.inf, -np.inf], np.nan)


# 中文说明：strict clinical_no_p4205 下的 11 个原始结构化字段按临床语义分组，
# 数值缺失指示与相同原字段绑定，避免 one-hot/缺失列被误当成独立变量。
STRUCTURED_VSN_GROUP_BY_COLUMN = {
    "p21022": "age",
    "age_squared": "age",
    "p20160_i0": "smoking_status",
    "p20162_i0": "smoking_exposure",
    "smoking_exposure_x_age": "smoking_exposure",
    "p3616_i0": "symptom_p3616",
    "p3751_i0": "symptom_p3751",
    "p4717_i0": "symptom_p4717",
    "p2316_i0": "symptom_p2316",
    "symptom_yes_count": "symptom_burden",
    "sex": "sex",
    # current 口径仍可运行；这些字段被单独归类为 P4205 摘要，正式 VSN 比较不使用该口径。
    "p21021_i0": "p4205_summary",
    "p4199_i0": "p4205_summary",
    "p4200_i0": "p4205_summary",
    "stiffness_x_age": "p4205_summary",
    "stiffness_per_age": "p4205_summary",
    "notch_shoulder_gap": "p4205_summary",
    "notch_shoulder_midpoint": "p4205_summary",
    "notch_position_ratio": "p4205_summary",
    "shoulder_position_ratio": "p4205_summary",
    "baseline_missing_count": "p4205_summary",
}


def build_vsn_group_indices(
    preprocessor: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
    column_to_group: dict[str, str],
    expected_dim: int,
) -> dict[str, list[int]]:
    """根据已拟合预处理器恢复编码列来源，生成严格覆盖的 VSN 分组索引。"""
    named = getattr(preprocessor, "named_transformers_", {})
    groups: dict[str, list[int]] = {}
    position = 0

    def add(column: str, indices: list[int]) -> None:
        group = column_to_group.get(column)
        if group is None:
            raise ValueError(f"VSN 未配置原始列 {column} 的变量组。")
        groups.setdefault(group, []).extend(indices)

    if numeric_columns:
        numeric_pipeline = named["num"]
        imputer = numeric_pipeline.named_steps["imputer"]
        for column in numeric_columns:
            add(column, [position])
            position += 1
        for source_idx in imputer.indicator_.features_.tolist():
            add(numeric_columns[int(source_idx)], [position])
            position += 1

    if categorical_columns:
        categorical_pipeline = named["cat"]
        encoder = categorical_pipeline.named_steps["onehot"]
        for column, categories in zip(categorical_columns, encoder.categories_, strict=True):
            width = int(len(categories))
            add(column, list(range(position, position + width)))
            position += width

    if position != int(expected_dim):
        raise ValueError(f"VSN 编码维度不一致：已分配 {position}，实际 {expected_dim}。")
    used = sorted(index for values in groups.values() for index in values)
    if used != list(range(int(expected_dim))):
        raise ValueError("VSN 分组未完整覆盖或重复覆盖编码列。")
    return groups


def build_identity_vsn_group_indices(preprocessor: object, columns: list[str], expected_dim: int) -> dict[str, list[int]]:
    """为 P4205 工程特征构建逐特征 VSN；缺失指示自动绑定到同一特征。"""
    return build_vsn_group_indices(
        preprocessor=preprocessor,
        numeric_columns=columns,
        categorical_columns=[],
        column_to_group={column: column for column in columns},
        expected_dim=expected_dim,
    )


def transform_features(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    p4205_cache: Path,
    structured_feature_set: str,
    p4205_cross_mode: str,
    cooperation_recipe: str = 'E0',
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """生成结构化、P4205 engineered 与 P4205-side cross feature 矩阵。"""
    structured = build_structured_frame(
        frame, feature_set=structured_feature_set)
    if structured_feature_set == "clinical_no_p4205_no_age":
        forbidden = {"p21022", "age_squared", "smoking_exposure_x_age"}
        leaked = sorted(forbidden.intersection(structured.columns))
        if leaked:
            raise ValueError(f"去年龄消融仍含年龄输入列：{leaked}")
    structured_preprocessor, numeric_columns, categorical_columns = build_tabular_preprocessor(
        structured.iloc[train_idx])
    x_train_structured = structured_preprocessor.fit_transform(
        structured.iloc[train_idx]).astype(np.float32, copy=False)
    x_val_structured = structured_preprocessor.transform(
        structured.iloc[val_idx]).astype(np.float32, copy=False)
    x_test_structured = structured_preprocessor.transform(
        structured.iloc[test_idx]).astype(np.float32, copy=False)
    structured_vsn_group_indices = build_vsn_group_indices(
        preprocessor=structured_preprocessor,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        column_to_group={**{c: c for c in structured.columns},
                         **STRUCTURED_VSN_GROUP_BY_COLUMN},
        expected_dim=int(x_train_structured.shape[1]),
    )
    structured_column_indices = build_vsn_group_indices(
        preprocessor=structured_preprocessor,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        column_to_group={column: column for column in structured.columns},
        expected_dim=int(x_train_structured.shape[1]),
    )
    p4205_frame = build_p4205_feature_tower_frame(
        frame[WAVE_COLUMN], eid_values=frame[EID_COLUMN], cache_path=p4205_cache)
    from cooperation_features import fit_selection
    selected_columns, cooperation_selection = fit_selection(
        p4205_frame, train_idx, cooperation_recipe)
    p4205_frame = p4205_frame.loc[:, selected_columns]
    print(f'[工程特征] {cooperation_recipe}: {cooperation_selection["original_count"]} → {len(selected_columns)}，仅训练分区拟合', flush=True)
    p4205_preprocessor = build_numeric_preprocessor(
        p4205_frame.iloc[train_idx])
    x_train_p4205 = p4205_preprocessor.fit_transform(
        p4205_frame.iloc[train_idx]).astype(np.float32, copy=False)
    x_val_p4205 = p4205_preprocessor.transform(
        p4205_frame.iloc[val_idx]).astype(np.float32, copy=False)
    x_test_p4205 = p4205_preprocessor.transform(
        p4205_frame.iloc[test_idx]).astype(np.float32, copy=False)
    p4205_vsn_group_indices = build_identity_vsn_group_indices(
        preprocessor=p4205_preprocessor,
        columns=p4205_frame.columns.tolist(),
        expected_dim=int(x_train_p4205.shape[1]),
    )
    if p4205_cross_mode == "selected_clinical_x_p4205":
        p4205_cross_frame = build_selected_p4205_cross_frame(
            structured, p4205_frame)
    elif p4205_cross_mode == "none":
        p4205_cross_frame = p4205_frame.copy()
    else:
        raise ValueError(f"未知 p4205_cross_mode：{p4205_cross_mode}")
    p4205_cross_preprocessor = build_numeric_preprocessor(
        p4205_cross_frame.iloc[train_idx])
    x_train_p4205_cross = p4205_cross_preprocessor.fit_transform(
        p4205_cross_frame.iloc[train_idx]).astype(np.float32, copy=False)
    x_val_p4205_cross = p4205_cross_preprocessor.transform(
        p4205_cross_frame.iloc[val_idx]).astype(np.float32, copy=False)
    x_test_p4205_cross = p4205_cross_preprocessor.transform(
        p4205_cross_frame.iloc[test_idx]).astype(np.float32, copy=False)
    p4205_cross_vsn_group_indices = build_identity_vsn_group_indices(
        preprocessor=p4205_cross_preprocessor,
        columns=p4205_cross_frame.columns.tolist(),
        expected_dim=int(x_train_p4205_cross.shape[1]),
    )
    matrices = {
        "structured_preprocessor": structured_preprocessor,
        "p4205_preprocessor": p4205_preprocessor,
        "p4205_cross_preprocessor": p4205_cross_preprocessor,
        "train_structured": x_train_structured,
        "val_structured": x_val_structured,
        "test_structured": x_test_structured,
        "train_p4205": x_train_p4205,
        "val_p4205": x_val_p4205,
        "test_p4205": x_test_p4205,
        "train_p4205_cross": x_train_p4205_cross,
        "val_p4205_cross": x_val_p4205_cross,
        "test_p4205_cross": x_test_p4205_cross,
        "structured_vsn_group_indices": structured_vsn_group_indices,
        "structured_column_indices": structured_column_indices,
        "p4205_vsn_group_indices": p4205_vsn_group_indices,
        "p4205_cross_vsn_group_indices": p4205_cross_vsn_group_indices,
    }
    metadata = {
        "cooperation_selection": cooperation_selection,
        "structured_feature_set": structured_feature_set,
        "structured_raw_feature_count": int(structured.shape[1]),
        "structured_raw_columns": structured.columns.tolist(),
        "structured_encoded_dim": int(x_train_structured.shape[1]),
        "p4205_cross_mode": p4205_cross_mode,
        "p4205_raw_feature_count": int(p4205_frame.shape[1]),
        "p4205_raw_columns": p4205_frame.columns.tolist(),
        "p4205_encoded_dim": int(x_train_p4205.shape[1]),
        "p4205_cross_raw_feature_count": int(p4205_cross_frame.shape[1]),
        "p4205_cross_encoded_dim": int(x_train_p4205_cross.shape[1]),
        "p4205_cross_added_feature_count": int(max(0, p4205_cross_frame.shape[1] - p4205_frame.shape[1])),
        "p4205_cross_clinical_columns": list(SELECTED_CLINICAL_CROSS_COLUMNS),
        "p4205_cross_p4205_columns": list(SELECTED_P4205_CROSS_COLUMNS),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "structured_vsn_group_names": list(structured_vsn_group_indices),
        "p4205_vsn_group_count": int(len(p4205_vsn_group_indices)),
        "p4205_cross_vsn_group_count": int(len(p4205_cross_vsn_group_indices)),
    }
    return matrices, metadata


def stable_model_seed(base_seed: int, model_name: str) -> int:
    """由基础 seed 与模型名生成稳定的独立 seed，消除模型列表顺序的影响。"""
    digest = hashlib.sha256(model_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((int(base_seed) + offset) % (2**31 - 1))


def effective_p4205_input(ablation_mode: str) -> str:
    """返回模型实际读取的 P4205 矩阵口径，避免把全局 cross 配置误当作模型输入。"""
    if ablation_mode in P4205_CROSS_ABLATION_MODES:
        return "selected_clinical_x_p4205"
    return "base_engineered_no_cross"


def select_p4205_input(
    ablation_mode: str,
    p4205: torch.Tensor,
    p4205_cross: torch.Tensor,
) -> torch.Tensor:
    """按模型模式选择实际张量；standard SAGM 永远返回基础 engineered 输入。"""
    return p4205_cross if ablation_mode in P4205_CROSS_ABLATION_MODES else p4205


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    """为每个模型的 DataLoader 使用独立 generator，保持 batch 顺序可复现。"""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator,
    )


def build_loss(
    loss_name: str,
    labels: np.ndarray,
    masks: np.ndarray,
    task_weights: np.ndarray,
    device: torch.device,
    focal_alpha: float,
    focal_gamma: float,
) -> nn.Module:
    """根据训练集标签构建多任务 loss。"""
    pos_weights: list[float] = []
    for task_index in range(labels.shape[1]):
        keep = masks[:, task_index] > 0
        positives = float(labels[keep, task_index].sum())
        negatives = float(keep.sum() - positives)
        pos_weights.append(negatives / max(1.0, positives))
    pos_weight = torch.as_tensor(
        pos_weights, dtype=torch.float32, device=device)
    weights = torch.as_tensor(
        task_weights[: labels.shape[1]], dtype=torch.float32, device=device)
    if loss_name == "weighted_bce":
        return MaskedWeightedBCEWithLogitsLoss(pos_weight, weights)
    return MaskedFocalBCEWithLogitsLoss(focal_alpha, focal_gamma, pos_weight, weights)


def output_to_2d(logits: torch.Tensor) -> torch.Tensor:
    return logits.unsqueeze(1) if logits.dim() == 1 else logits


def train_one_epoch(
    model: SAGMNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    gradient_totals = {}
    from training_diagnostics import GradientTracker
    tracker = GradientTracker()
    batch_count = 0
    for batch_index, (wave, p4205, p4205_cross, structured, labels, masks) in enumerate(loader, 1):
        wave = wave.to(device) if model.wave_tower is not None else wave
        p4205 = p4205.to(device)
        p4205_cross = p4205_cross.to(device)
        p4205_input = select_p4205_input(
            model.ablation_mode, p4205, p4205_cross)
        structured = structured.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.regularization_control.before_step()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(model, 'cooperation_loss'):
            loss = model.cooperation_loss(
                wave, p4205_input, structured, labels, masks, criterion)
        else:
            logits = output_to_2d(model(wave, p4205_input, structured))
            loss = criterion(logits, labels, masks)
        loss.backward()
        if getattr(model, 'collect_cooperation_diagnostics', False):
            from cooperation_model import gradient_rms
            for key, value in gradient_rms(model).items():
                gradient_totals[key] = gradient_totals.get(key, 0.0) + value
            batch_count += 1
        tracker.observe(model.parameters(), grad_clip_norm)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        model.regularization_control.after_step(model)
        callback = getattr(model, "diagnostic_midpoint", None)
        if callback is not None and batch_index == (len(loader)+1)//2:
            callback(batch_index)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    model.regularization_gradient_diagnostics = tracker.summary()
    model.last_gradient_diagnostics = {
        key: value / max(1, batch_count) for key, value in gradient_totals.items()}
    return total_loss / max(1, total_count)


@torch.no_grad()
def collect_predictions(
    model: SAGMNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    control = getattr(model, "regularization_control", None)
    if control is not None:
        model = control.evaluated(model)
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    aux_rows: dict[str, list[np.ndarray]] = {
        "gate_wave": [],
        "gate_feature": [],
        "structured_embedding_norm": [],
        "wave_embedding_norm": [],
        "p4205_feature_embedding_norm": [],
    }
    if hasattr(model, 'cooperation'):
        aux_rows.update(wave_contribution_norm=[],
                        feature_contribution_norm=[])
    vector_gate_rows: dict[str, list[np.ndarray]] = {
        "gate_wave_vector": [], "gate_feature_vector": []}
    weight_rows: dict[str, list[np.ndarray]] = {
        "structured_vsn_weights": [], "p4205_vsn_weights": []}
    total_loss = 0.0
    total_count = 0
    for wave, p4205, p4205_cross, structured, labels, masks in loader:
        wave = wave.to(device) if model.wave_tower is not None else wave
        p4205 = p4205.to(device)
        p4205_cross = p4205_cross.to(device)
        p4205_input = select_p4205_input(
            model.ablation_mode, p4205, p4205_cross)
        structured = structured.to(device)
        labels_device = labels.to(device)
        masks_device = masks.to(device)
        logits, aux = model(wave, p4205_input, structured, return_aux=True)
        logits = output_to_2d(logits)
        loss = criterion(logits, labels_device, masks_device)
        probability = torch.sigmoid(logits)
        all_logits.append(probability.detach().cpu().numpy())
        all_labels.append(labels.numpy())
        all_masks.append(masks.numpy())
        for key in aux_rows:
            aux_rows[key].append(aux[key].detach().cpu().numpy())
        for key in vector_gate_rows:
            vector_gate = aux[key]
            if vector_gate.ndim == 2 and vector_gate.shape[1] > 0:
                vector_gate_rows[key].append(
                    vector_gate.detach().cpu().numpy())
        for key in weight_rows:
            weights = aux[key]
            if weights.shape[1] > 0:
                weight_rows[key].append(weights.detach().cpu().numpy())
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    result: dict[str, np.ndarray | float] = {
        "loss": total_loss / max(1, total_count),
        "probability": np.concatenate(all_logits, axis=0),
        "label": np.concatenate(all_labels, axis=0),
        "mask": np.concatenate(all_masks, axis=0),
    }
    for key, values in aux_rows.items():
        result[key] = np.concatenate(values, axis=0)
    for key, values in vector_gate_rows.items():
        if values:
            result[key] = np.concatenate(values, axis=0)
    for key, values in weight_rows.items():
        if values:
            result[key] = np.concatenate(values, axis=0)
    return result


def task_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    threshold: float | None = None,
    threshold_source: str = "current_split",
) -> dict[str, float | str | list[list[int]] | None]:
    """计算单任务指标；阈值可由 validation 传入，避免 test label 参与阈值选择。"""
    keep = mask > 0
    y = y_true[keep].astype(np.int64)
    p = probability[keep].astype(np.float64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "positive_count": int(y.sum()),
            "positive_rate": float(y.mean()) if y.size else None,
            "n": int(y.size),
            "threshold": float(threshold) if threshold is not None else None,
            "threshold_source": threshold_source if threshold is not None else "unavailable",
        }
    if threshold is None:
        threshold = select_threshold(y, p)
    pred = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | str | list[list[int]] | None] = {
        "n": int(y.size),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    metrics.update(topk_metrics(y, p))
    return metrics


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    thresholds = np.unique(np.quantile(
        probability, np.linspace(0.05, 0.95, 91)))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in thresholds:
        pred = (probability >= threshold).astype(np.int8)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def select_task_thresholds(pred: dict[str, np.ndarray | float], tasks: list[str]) -> dict[str, float | None]:
    """只在 validation predictions 上选择各任务阈值，供 validation/test 统一复用。"""
    labels = pred["label"]
    probabilities = pred["probability"]
    masks = pred["mask"]
    assert isinstance(labels, np.ndarray) and isinstance(
        probabilities, np.ndarray) and isinstance(masks, np.ndarray)
    thresholds: dict[str, float | None] = {}
    for index, task in enumerate(tasks):
        keep = masks[:, index] > 0
        y = labels[keep, index].astype(np.int64)
        p = probabilities[keep, index].astype(np.float64)
        thresholds[task] = float(select_threshold(
            y, p)) if y.size and len(np.unique(y)) >= 2 else None
    return thresholds


def evaluate_payload(
    pred: dict[str, np.ndarray | float],
    tasks: list[str],
    thresholds: dict[str, float | None] | None = None,
    threshold_source: str = "current_split",
) -> dict[str, object]:
    labels = pred["label"]
    probabilities = pred["probability"]
    masks = pred["mask"]
    assert isinstance(labels, np.ndarray) and isinstance(
        probabilities, np.ndarray) and isinstance(masks, np.ndarray)
    task_payload = {}
    thresholds_were_provided = thresholds is not None
    thresholds = thresholds or {}
    for index, task in enumerate(tasks):
        task_threshold = thresholds.get(task)
        local_source = threshold_source if (
            task_threshold is not None or not thresholds_were_provided) else "unavailable"
        task_payload[task] = task_metrics(
            labels[:, index],
            probabilities[:, index],
            masks[:, index],
            threshold=task_threshold,
            threshold_source=local_source,
        )
    return {"loss": float(pred["loss"]), "tasks": task_payload}


def compute_early_stop_score(all_cause_metrics: dict[str, object], metric_name: str) -> float:
    """把早停指标集中到一个函数，便于单测和 ROC 导向复验。"""
    pr_auc = float(all_cause_metrics.get("pr_auc") or 0.0)
    roc_auc = float(all_cause_metrics.get("roc_auc") or 0.0)
    if metric_name == "pr_auc":
        return pr_auc
    if metric_name == "roc_auc":
        return roc_auc
    if metric_name == "mean_pr_roc":
        return 0.5 * (pr_auc + roc_auc)
    raise ValueError(f"Unsupported early stop metric: {metric_name}")


def early_stop_snapshot(
    payload: dict[str, object], tasks: list[str], aggregation: str, metric_name: str
) -> tuple[float, float, float]:
    """返回早停分数及其 PR/ROC 审计值；诊断实验对有效输出头做宏平均。"""
    task_payload = payload["tasks"]
    assert isinstance(task_payload, dict)
    if aggregation == "landmark_all_cause":
        selected_tasks = ("all_cause_0_5", "all_cause_5_10")
        selected = [task_payload[name] for name in selected_tasks]
        pr_auc = float(
            np.mean([float(item.get("pr_auc") or 0.0) for item in selected]))
        roc_auc = float(
            np.mean([float(item.get("roc_auc") or 0.0) for item in selected]))
        score = compute_early_stop_score(
            {"pr_auc": pr_auc, "roc_auc": roc_auc}, metric_name)
        return score, pr_auc, roc_auc
    if aggregation == "all_cause":
        selected = task_payload["all_cause"]
        assert isinstance(selected, dict)
        pr_auc = float(selected.get("pr_auc") or 0.0)
        roc_auc = float(selected.get("roc_auc") or 0.0)
        return compute_early_stop_score(selected, metric_name), pr_auc, roc_auc
    if aggregation != "macro_tasks":
        raise ValueError(f"未知 early-stop aggregation：{aggregation}")
    pr_values: list[float] = []
    roc_values: list[float] = []
    scores: list[float] = []
    for task in tasks:
        selected = task_payload[task]
        assert isinstance(selected, dict)
        if selected.get("pr_auc") is None or selected.get("roc_auc") is None:
            continue
        pr_values.append(float(selected["pr_auc"]))
        roc_values.append(float(selected["roc_auc"]))
        scores.append(compute_early_stop_score(selected, metric_name))
    if not scores:
        return 0.0, 0.0, 0.0
    return float(np.mean(scores)), float(np.mean(pr_values)), float(np.mean(roc_values))


def train_model(
    model: SAGMNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    tasks: list[str],
    early_stop_aggregation: str,
) -> tuple[dict[str, object], dict[str, np.ndarray | float], dict[str, np.ndarray | float], pd.DataFrame]:
    """训练单个 SAGM 模型；仅诊断实验改变早停聚合，不改变网络结构。"""
    optimizer = torch.optim.AdamW(
        (x for x in model.parameters() if x.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    from regularization_control import Controller, unweighted_bce
    control = Controller(model, optimizer, len(train_loader), getattr(args, "regularization_mode", "none"))
    model.regularization_control = control
    best_state = copy.deepcopy(model.state_dict())
    initial_state = copy.deepcopy(best_state) if getattr(
        model, 'include_epoch_zero', False) else None
    best_score = -math.inf
    best_epoch = 0
    best_val_pr_auc = 0.0
    best_val_roc_auc = 0.0
    stale_epochs = 0
    history_rows: list[dict[str, object]] = []
    # 用独立、顺序固定的eval loader诊断训练表现，不消耗训练shuffle的随机状态。
    diagnostic_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   generator=torch.Generator().manual_seed(9173))
    started = time.time()
    training_compute_seconds = validation_seconds = 0.0
    if getattr(model, 'include_epoch_zero', False):
        zero_pred = collect_predictions(model, val_loader, criterion, device)
        zero_payload = evaluate_payload(
            zero_pred, tasks, threshold_source='validation_epoch_zero')
        best_score, best_val_pr_auc, best_val_roc_auc = early_stop_snapshot(
            zero_payload, tasks, early_stop_aggregation, args.early_stop_metric)
        history_rows.append(dict(epoch=0, val_loss=zero_pred['loss'], val_pr_auc=best_val_pr_auc,
                                 val_roc_auc=best_val_roc_auc, best_epoch=0, early_stop_score=best_score))
        validation_seconds += time.time() - started
        print(
            f'[epoch0] 冻结anchor S={best_score:.8f}，作为可回退最佳checkpoint', flush=True)
    from training_diagnostics import record_prediction, isolated_evaluation
    diagnostic_dir = args.output_dir / "epoch_diagnostics"
    for epoch in range(1, args.epochs + 1):
        if getattr(args, "diagnostic_half_epoch", False):
            def midpoint(batch_index):
                with isolated_evaluation(model):
                    pred = collect_predictions(model, val_loader, criterion, device)
                    record_prediction(diagnostic_dir, epoch-1+batch_index/len(train_loader), (epoch-1)*len(train_loader)+batch_index, pred, criterion)
            model.diagnostic_midpoint = midpoint
        control.lrs = []
        phase_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip_norm)
        training_compute_seconds += time.time() - phase_start
        phase_start = time.time()
        with torch.no_grad():
            train_pred = collect_predictions(
                model, diagnostic_loader, criterion, device)
        train_payload = evaluate_payload(
            train_pred, tasks, threshold_source="training_diagnostic_only")
        train_score, train_pr, train_roc = early_stop_snapshot(
            train_payload, tasks, early_stop_aggregation, args.early_stop_metric)
        val_pred = collect_predictions(model, val_loader, criterion, device)
        val_payload = evaluate_payload(
            val_pred, tasks, threshold_source="validation_current_epoch")
        record_prediction(diagnostic_dir, epoch, epoch*len(train_loader), val_pred, criterion)
        score, val_pr_auc, val_roc_auc = early_stop_snapshot(
            val_payload, tasks, early_stop_aggregation, args.early_stop_metric
        )
        validation_seconds += time.time() - phase_start
        elapsed = time.time() - started
        mean_epoch = elapsed / max(1, epoch)
        eta = mean_epoch * max(0, args.epochs - epoch)
        improved = score > best_score + args.min_delta
        if improved:
            best_state = copy.deepcopy(control.evaluated(model).state_dict())
            best_score = score
            best_epoch = epoch
            best_val_pr_auc = val_pr_auc
            best_val_roc_auc = val_roc_auc
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "optimizer_steps": epoch*len(train_loader),
            "steps_per_epoch": len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rate_min": min(control.lrs),
            "learning_rate_mean": float(np.mean(control.lrs)),
            "learning_rate_max": max(control.lrs),
            "train_unweighted_bce": unweighted_bce(train_pred),
            "val_unweighted_bce": unweighted_bce(val_pred),
            **model.regularization_gradient_diagnostics,
            "train_eval_loss": train_payload['loss'],
            "train_pr_auc": train_pr,
            "train_roc_auc": train_roc,
            "train_early_stop_score": train_score,
            "val_loss": val_payload["loss"],
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc,
            "early_stop_metric": args.early_stop_metric,
            "early_stop_aggregation": early_stop_aggregation,
            "early_stop_score": score,
            "best_epoch": best_epoch,
            "best_early_stop_score": best_score,
            "best_val_pr_auc": best_val_pr_auc,
            "best_val_roc_auc": best_val_roc_auc,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
        }
        if control.ema is not None:
            with isolated_evaluation(model):
                control.active = False
                try:
                    raw_train = collect_predictions(model, diagnostic_loader, criterion, device)
                    raw_val = collect_predictions(model, val_loader, criterion, device)
                finally:
                    control.active = True
            row['online_train_eval_loss'] = raw_train['loss']
            row['online_val_loss'] = raw_val['loss']
            row['online_val_unweighted_bce'] = unweighted_bce(raw_val)
            for prefix, pred in [('online_train',raw_train),('online_val',raw_val)]:
                pay = evaluate_payload(pred, tasks, threshold_source='diagnostic_only')
                _, ap, roc = early_stop_snapshot(pay,tasks,early_stop_aggregation,args.early_stop_metric)
                row[prefix+'_pr_auc'] = ap
                row[prefix+'_roc_auc'] = roc
        row.update(getattr(model, 'last_gradient_diagnostics', {}))
        for key in ('gate_wave', 'gate_feature', 'wave_contribution_norm', 'feature_contribution_norm'):
            if key in val_pred:
                for q, suffix in ((0.25, 'q25'), (0.5, 'median'), (0.75, 'q75')):
                    row[f'val_{key}_{suffix}'] = float(
                        np.quantile(val_pred[key], q))
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(args.output_dir / "live_history.csv", index=False)
        print(
            f"[SAGM] epoch={epoch}/{args.epochs} train_loss={train_loss:.5f} "
            f"val_pr_auc={val_pr_auc:.5f} val_roc_auc={val_roc_auc:.5f} "
            f"early_stop={early_stop_aggregation}/{args.early_stop_metric} score={score:.5f} "
            f"best_epoch={best_epoch} elapsed_min={elapsed / 60:.1f} eta_min={eta / 60:.1f}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(
                f"[SAGM] early stop at epoch={epoch}, best_epoch={best_epoch}, "
                f"metric={args.early_stop_metric}, best_score={best_score:.5f}",
                flush=True,
            )
            break
    if initial_state is not None:
        final_state = model.state_dict()
        frozen_keys = [k for k in initial_state if k.startswith('anchor.')]
        if not all(torch.equal(initial_state[k], final_state[k]) for k in frozen_keys):
            raise AssertionError('训练期间冻结anchor的参数或BN缓冲发生变化')
        changed = [name for name, parameter in model.named_parameters()
                   if parameter.requires_grad and not torch.equal(initial_state[name], parameter)]
        if not any(name.startswith('increment_tower.') for name in changed) or not any(name.startswith('delta_head.') for name in changed):
            raise AssertionError('训练后新增塔或增量头没有有效更新')
        model.frozen_training_audit = dict(
            anchor_exact_unchanged=True, updated_parameters=changed)
    control.active = False
    model.load_state_dict(best_state)
    phase_start = time.time()
    val_pred = collect_predictions(model, val_loader, criterion, device)
    validation_only = bool(getattr(args, 'cooperation_validation_only', False))
    test_pred = None if validation_only else collect_predictions(
        model, test_loader, criterion, device)
    validation_thresholds = select_task_thresholds(val_pred, tasks)
    validation_payload = evaluate_payload(
        val_pred, tasks, thresholds=validation_thresholds, threshold_source="validation"
    )
    test_payload = None if validation_only else evaluate_payload(
        test_pred, tasks, thresholds=validation_thresholds, threshold_source="validation"
    )
    selected_score, selected_pr_auc, selected_roc_auc = early_stop_snapshot(
        validation_payload, tasks, early_stop_aggregation, args.early_stop_metric
    )
    payload = {
        "best_epoch": int(best_epoch),
        "early_stop_metric": args.early_stop_metric,
        "early_stop_aggregation": early_stop_aggregation,
        "best_early_stop_score": float(selected_score),
        "best_val_pr_auc": float(selected_pr_auc),
        "best_val_roc_auc": float(selected_roc_auc),
        "thresholds": validation_thresholds,
        "threshold_source": "validation",
        "validation": validation_payload,
        "test": test_payload,
        "training_seconds": float(time.time() - started),
        "epochs_ran": int(sum(row['epoch'] > 0 for row in history_rows)),
        "training_compute_seconds": training_compute_seconds,
        "validation_seconds": validation_seconds + time.time() - phase_start,
        "epoch_zero_candidate": bool(getattr(model, 'include_epoch_zero', False)),
        "frozen_training_audit": getattr(model, 'frozen_training_audit', None),
    }
    if validation_only:
        payload.pop('test', None)
    return payload, val_pred, test_pred, pd.DataFrame(history_rows)


def save_prediction_table(
    path: Path,
    eid_values: np.ndarray,
    pred: dict[str, np.ndarray | float],
    tasks: list[str],
) -> None:
    probabilities = pred["probability"]
    labels = pred["label"]
    masks = pred["mask"]
    assert isinstance(probabilities, np.ndarray) and isinstance(
        labels, np.ndarray) and isinstance(masks, np.ndarray)
    frame = pd.DataFrame({EID_COLUMN: eid_values})
    for index, task in enumerate(tasks):
        frame[f"{task}_true"] = safe_binary_int_array(labels[:, index])
        frame[f"{task}_mask"] = safe_binary_int_array(masks[:, index])
        frame[f"{task}_probability"] = probabilities[:, index]
    for key in [
        "gate_wave",
        "gate_feature",
        "structured_embedding_norm",
        "wave_embedding_norm",
        "p4205_feature_embedding_norm",
    ]:
        value = pred[key]
        assert isinstance(value, np.ndarray)
        frame[key] = value
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_vsn_weight_artifacts(
    model_dir: Path,
    eid_values: np.ndarray,
    pred: dict[str, np.ndarray | float],
    model: SAGMNet,
) -> None:
    """导出 VSN 权重：结构化字段用 CSV，P4205 高维权重使用压缩矩阵和汇总表。"""
    labels = pred["label"]
    probabilities = pred["probability"]
    masks = pred["mask"]
    assert isinstance(labels, np.ndarray) and isinstance(
        probabilities, np.ndarray) and isinstance(masks, np.ndarray)
    label = safe_binary_int_array(labels[:, 0])
    probability = probabilities[:, 0].astype(np.float32, copy=False)
    mask = safe_binary_int_array(masks[:, 0])
    exports = (
        ("structured", "structured_vsn_weights", getattr(
            model.structured_tower, "group_names", ())),
        ("p4205", "p4205_vsn_weights", getattr(
            model.p4205_feature_tower, "group_names", ())),
    )
    for prefix, key, names in exports:
        weights = pred.get(key)
        if not isinstance(weights, np.ndarray) or weights.ndim != 2 or weights.shape[1] == 0:
            continue
        group_names = [str(name) for name in names]
        if len(group_names) != weights.shape[1]:
            raise ValueError(f"{prefix} VSN 权重列数与组名数不一致。")
        if prefix == "structured":
            frame = pd.DataFrame({
                EID_COLUMN: eid_values,
                "all_cause_true": label,
                "all_cause_mask": mask,
                "all_cause_probability": probability,
            })
            for index, name in enumerate(group_names):
                frame[f"vsn_weight__{name}"] = weights[:, index]
            frame.to_csv(model_dir / "structured_vsn_weights_test.csv",
                         index=False, encoding="utf-8-sig")
            continue

        np.savez_compressed(
            model_dir / "p4205_vsn_weights_test.npz",
            eid=np.asarray(eid_values),
            all_cause_true=label,
            all_cause_mask=mask,
            all_cause_probability=probability,
            group_names=np.asarray(group_names, dtype=str),
            weights=weights.astype(np.float32, copy=False),
        )
        positive = (label == 1) & (mask == 1)
        negative = (label == 0) & (mask == 1)
        topk = min(10, weights.shape[1])
        cutoff = np.partition(weights, -topk, axis=1)[:, [-topk]]
        summary = pd.DataFrame({
            "feature": group_names,
            "mean_weight": weights.mean(axis=0),
            "std_weight": weights.std(axis=0),
            "positive_mean_weight": weights[positive].mean(axis=0) if positive.any() else np.nan,
            "negative_mean_weight": weights[negative].mean(axis=0) if negative.any() else np.nan,
            "top10_frequency": (weights >= cutoff).mean(axis=0),
        })
        summary.sort_values("mean_weight", ascending=False).to_csv(
            model_dir / "p4205_vsn_weight_summary.csv", index=False, encoding="utf-8-sig"
        )


def save_vector_gate_artifacts(
    model_dir: Path,
    eid_values: np.ndarray,
    pred: dict[str, np.ndarray | float],
) -> None:
    """导出向量 gate 的维度汇总与样本级强度，不把 128 维门值混入通用预测表。"""
    wave_gate = pred.get("gate_wave_vector")
    feature_gate = pred.get("gate_feature_vector")
    if not isinstance(wave_gate, np.ndarray) or not isinstance(feature_gate, np.ndarray):
        return
    if wave_gate.ndim != 2 or feature_gate.ndim != 2 or wave_gate.shape[1] == 0 or feature_gate.shape[1] == 0:
        return
    if wave_gate.shape[0] != len(eid_values) or feature_gate.shape[0] != len(eid_values):
        raise ValueError("向量 gate 审计行数与测试 EID 不一致。")
    labels = pred["label"]
    probabilities = pred["probability"]
    masks = pred["mask"]
    assert isinstance(labels, np.ndarray) and isinstance(
        probabilities, np.ndarray) and isinstance(masks, np.ndarray)
    sample = pd.DataFrame(
        {
            EID_COLUMN: eid_values,
            "all_cause_true": safe_binary_int_array(labels[:, 0]),
            "all_cause_mask": safe_binary_int_array(masks[:, 0]),
            "all_cause_probability": probabilities[:, 0],
            "wave_gate_l1": np.abs(wave_gate).sum(axis=1),
            "wave_gate_l2": np.linalg.vector_norm(wave_gate, axis=1),
            "engineered_gate_l1": np.abs(feature_gate).sum(axis=1),
            "engineered_gate_l2": np.linalg.vector_norm(feature_gate, axis=1),
        }
    )
    sample.to_csv(model_dir / "vector_gate_sample_strength_test.csv",
                  index=False, encoding="utf-8-sig")
    summaries = []
    for tower, values in (("waveform", wave_gate), ("engineered_p4205", feature_gate)):
        summaries.append(
            pd.DataFrame(
                {
                    "tower": tower,
                    "dimension": np.arange(values.shape[1], dtype=int),
                    "mean_gate": values.mean(axis=0),
                    "std_gate": values.std(axis=0),
                }
            )
        )
    pd.concat(summaries, ignore_index=True).to_csv(
        model_dir / "vector_gate_dimension_summary.csv", index=False, encoding="utf-8-sig"
    )


def flatten_metrics(model_name: str, encoder: str, payload: dict[str, object], split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    split_payload = payload[split]
    for task, metrics in split_payload["tasks"].items():
        rows.append(
            {
                "model_name": model_name,
                "wave_encoder": encoder,
                "split": split,
                "task": task,
                "best_epoch": payload["best_epoch"],
                "early_stop_metric": payload.get("early_stop_metric"),
                "best_early_stop_score": payload.get("best_early_stop_score"),
                "best_val_pr_auc": payload.get("best_val_pr_auc"),
                "best_val_roc_auc": payload.get("best_val_roc_auc"),
                "parameter_count": payload.get("parameter_count"),
                "training_seconds": payload.get("training_seconds"),
                "epochs_ran": payload.get("epochs_ran"),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            }
        )
    return rows


def bootstrap_metric_ci(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    repeats: int,
    seed: int,
    comparison: str = "sagm_gated_vs_structured_only",
) -> pd.DataFrame:
    """配对 bootstrap 比较模型 B 相对模型 A 的指标差异。"""
    rng = np.random.default_rng(seed)
    rows = []
    y_true = safe_binary_int_array(y_true)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    n = len(y_true)
    for metric_name, metric_fn in {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "top_5pct_precision": lambda y, p: topk_metrics(y, p)["top_5pct_precision"],
    }.items():
        estimates = []
        for _ in range(repeats):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            estimates.append(
                float(metric_fn(y_true[idx], p_b[idx]) - metric_fn(y_true[idx], p_a[idx])))
        if estimates:
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric_name,
                    "delta": float(metric_fn(y_true, p_b) - metric_fn(y_true, p_a)),
                    "ci_low": float(np.percentile(estimates, 2.5)),
                    "ci_high": float(np.percentile(estimates, 97.5)),
                    "bootstrap_repeats": int(len(estimates)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_prediction_pair_ci(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    comparison: str,
    repeats: int,
    seed: int,
    task: str = "all_cause",
) -> pd.DataFrame:
    """按 eid 对齐两个预测表，再做同一测试样本上的配对 bootstrap。"""
    columns = [EID_COLUMN, f"{task}_true",
               f"{task}_mask", f"{task}_probability"]
    missing = [
        column for column in columns if column not in baseline.columns or column not in challenger.columns]
    if missing:
        return pd.DataFrame()
    left = baseline[columns].copy()
    right = challenger[columns].copy()
    merged = left.merge(right, on=EID_COLUMN, suffixes=(
        "_a", "_b"), validate="one_to_one")
    keep = (merged[f"{task}_mask_a"].to_numpy(dtype=float) > 0) & (
        merged[f"{task}_mask_b"].to_numpy(dtype=float) > 0)
    if keep.sum() < 2:
        return pd.DataFrame()
    y_a = safe_binary_int_array(merged.loc[keep, f"{task}_true_a"].to_numpy())
    y_b = safe_binary_int_array(merged.loc[keep, f"{task}_true_b"].to_numpy())
    if not np.array_equal(y_a, y_b) or len(np.unique(y_a)) < 2:
        return pd.DataFrame()
    result = bootstrap_metric_ci(
        y_a,
        merged.loc[keep, f"{task}_probability_a"].to_numpy(dtype=float),
        merged.loc[keep, f"{task}_probability_b"].to_numpy(dtype=float),
        repeats,
        seed,
        comparison=comparison,
    )
    if not result.empty:
        result.insert(1, "task", task)
        result.insert(2, "n", int(keep.sum()))
        result.insert(3, "positive_count", int(y_a.sum()))
    return result


def build_tower_bootstrap_ci(
    predictions: dict[str, pd.DataFrame],
    wave_encoders: list[str],
    repeats: int,
    seed: int,
    tasks: list[str] | tuple[str, ...] = ("all_cause",),
) -> pd.DataFrame:
    """生成塔级消融的配对 bootstrap 表，用于判断 raw/engineered/cross 的独立增量。"""
    comparison_specs: list[tuple[str, str, str]] = []
    if "structured_only" in predictions:
        for encoder in wave_encoders:
            comparison_specs.extend(
                [
                    ("structured_only", f"sagm_gated_raw_only_{encoder}",
                     f"sagm_gated_raw_only_{encoder}_vs_structured_only"),
                    ("structured_only", f"two_tower_raw_{encoder}",
                     f"two_tower_raw_{encoder}_vs_structured_only"),
                    ("structured_only", f"three_tower_concat_{encoder}",
                     f"three_tower_concat_{encoder}_vs_structured_only"),
                    ("structured_only", f"sagm_gated_{encoder}",
                     f"sagm_gated_{encoder}_vs_structured_only"),
                    ("sagm_gated_feature_only",
                     f"sagm_gated_{encoder}", f"sagm_gated_{encoder}_vs_sagm_gated_feature_only"),
                    (f"sagm_gated_raw_only_{encoder}", f"sagm_gated_{encoder}",
                     f"sagm_gated_{encoder}_vs_sagm_gated_raw_only_{encoder}"),
                ]
            )
        comparison_specs.extend(
            [
                ("structured_only", "sagm_gated_feature_only",
                 "sagm_gated_feature_only_vs_structured_only"),
                ("structured_only", "structured_engineered",
                 "structured_engineered_vs_structured_only"),
                ("structured_engineered", "structured_p4205_cross",
                 "structured_p4205_cross_vs_structured_engineered"),
                ("sagm_gated_feature_only", "sagm_gated_cross_feature",
                 "sagm_gated_cross_feature_vs_sagm_gated_feature_only"),
                ("sagm_gated_inception_lite", "sagm_gated_cross_three_tower_inception_lite",
                 "sagm_gated_cross_three_tower_vs_sagm_gated_inception_lite"),
            ]
        )
    rows: list[pd.DataFrame] = []
    for baseline_name, challenger_name, comparison in comparison_specs:
        if baseline_name not in predictions or challenger_name not in predictions:
            continue
        for task in tasks:
            local = bootstrap_prediction_pair_ci(
                predictions[baseline_name],
                predictions[challenger_name],
                comparison=comparison,
                repeats=repeats,
                seed=seed,
                task=task,
            )
            if not local.empty:
                local.insert(1, "baseline_model", baseline_name)
                local.insert(2, "challenger_model", challenger_name)
                rows.append(local)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def gate_interpretability(
    test_meta: pd.DataFrame,
    prediction: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """生成 gate 分组、死因和相关性解释表。"""
    frame = test_meta.reset_index(drop=True).copy()
    gate_cols = ["gate_wave", "gate_feature"]
    frame[gate_cols] = prediction[gate_cols].to_numpy()
    frame["age_group"] = pd.qcut(pd.to_numeric(
        frame[AGE_COLUMN], errors="coerce"), q=4, duplicates="drop")
    frame["stiffness_group"] = pd.qcut(pd.to_numeric(
        frame["p21021_i0"], errors="coerce"), q=4, duplicates="drop")
    rows: list[dict[str, object]] = []
    group_specs = {
        "all": pd.Series("all", index=frame.index),
        "age_group": frame["age_group"].astype("string"),
        "stiffness_group": frame["stiffness_group"].astype("string"),
        "sex": frame["sex"].astype("string"),
        "all_cause": frame["target"].map({0: "negative", 1: "positive"}),
        "circulatory_I": frame["icd10_chapter"].eq("I").map({False: "negative", True: "positive"}),
        "respiratory_J": frame["icd10_chapter"].eq("J").map({False: "negative", True: "positive"}),
        "malignant_neoplasm_C": frame["icd10_chapter"].eq("C").map({False: "negative", True: "positive"}),
    }
    for group_name, values in group_specs.items():
        local = frame.assign(group_value=values)
        for group_value, part in local.groupby("group_value", dropna=False):
            for gate in gate_cols:
                rows.append(
                    {
                        "analysis": "group_summary",
                        "group": group_name,
                        "group_value": str(group_value),
                        "metric": gate,
                        "n": int(len(part)),
                        "mean": float(part[gate].mean()),
                        "median": float(part[gate].median()),
                        "std": float(part[gate].std(ddof=0)),
                    }
                )
    for variable in [AGE_COLUMN, "p21021_i0", "p4199_i0", "p4200_i0"]:
        numeric = pd.to_numeric(frame[variable], errors="coerce")
        for gate in gate_cols:
            rows.append(
                {
                    "analysis": "spearman",
                    "group": variable,
                    "group_value": "all",
                    "metric": gate,
                    "n": int(numeric.notna().sum()),
                    "mean": float(numeric.corr(frame[gate], method="spearman")),
                    "median": np.nan,
                    "std": np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "gate_interpretability_summary.csv",
                   index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    args = parse_args()
    train_seed = int(
        args.train_seed) if args.train_seed is not None else int(args.seed)
    set_global_seed(train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = resolve_sagm_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(args.data, args.target_years, args.max_rows)
    frame = add_endpoint_columns(
        frame, args.data, args.max_rows, args.task_mode)
    landmark_bundle = None
    landmark_audit = None
    if args.task_mode == "landmark_death_mtl":
        if args.followup_dates is None:
            raise ValueError(
                "landmark_death_mtl 必须提供 --followup-dates datasets/followup_dates.xlsx。")
        # 日期队列和窗口已在建库阶段统一冻结，禁止旧函数再次排除截止日后死亡者。
        landmark_bundle = load_followup_workbook(args.followup_dates)
        landmark_audit = pd.DataFrame(
            {'reason': ['frozen_exact_date_cohort'], 'count': [len(frame)]})
    frame = apply_cohort_filter(frame, args.cohort, args.p4205_cache)
    tasks, labels, masks = build_multitask_labels(frame, args.task_mode)
    early_stop_aggregation = resolve_early_stop_aggregation(
        args.task_mode, args.early_stop_aggregation)
    if early_stop_aggregation == "all_cause" and "all_cause" not in tasks:
        raise ValueError("all_cause 早停只适用于包含全因死亡头的任务模式。")
    if args.split_assignments is not None:
        split_indices = load_frozen_split_indices(
            frame, args.split_assignments)
        split_source = str(args.split_assignments.resolve())
    else:
        split_indices = build_generated_split_indices(
            labels, masks, tasks, args.task_mode, args.test_size, args.val_size, args.seed
        )
        split_source = f"generated_hierarchy_seed_{args.seed}"
    train_idx = split_indices["train"]
    val_idx = split_indices["validation"]
    test_idx = split_indices["test"]
    if landmark_bundle is not None and landmark_audit is not None:
        landmark_audit.to_csv(
            output_dir / "landmark_followup_audit.csv", index=False, encoding="utf-8-sig")
        (output_dir / "landmark_followup_metadata.json").write_text(
            json.dumps(landmark_bundle.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    split_summary = save_split_diagnostics(
        output_dir, frame, split_indices, tasks, labels, masks)
    age_split_distribution(frame, split_indices, labels, masks, tasks).to_csv(
        output_dir / "age_subgroup_split_distribution.csv", index=False, encoding="utf-8-sig"
    )
    if any(mode in P4205_CROSS_ABLATION_MODES for mode in args.models) and args.p4205_cross_mode == "none":
        raise ValueError(
            "运行 P4205 cross 消融时必须指定 --p4205-cross-mode selected_clinical_x_p4205。")
    matrices, feature_meta = transform_features(
        frame, train_idx, val_idx, test_idx, args.p4205_cache, args.structured_feature_set, args.p4205_cross_mode
    )
    import joblib
    joblib.dump(matrices['structured_preprocessor'],
                output_dir/'structured_preprocessor.joblib')
    joblib.dump(matrices['p4205_preprocessor'],
                output_dir/'p4205_preprocessor.joblib')
    joblib.dump(matrices['p4205_cross_preprocessor'],
                output_dir/'p4205_cross_preprocessor.joblib')
    # 导出原始结构化列在本次策略下的实际保留/剔除结果，供去年龄消融审计。
    candidate_columns = build_structured_frame(
        frame, feature_set="current").columns.tolist()
    policy = split_structured_features(
        candidate_columns, args.structured_feature_set)
    policy_table = pd.DataFrame({
        "feature": candidate_columns,
        "kept": [column in set(policy.kept_features) for column in candidate_columns],
        "dropped": [column in set(policy.dropped_features) for column in candidate_columns],
        "age_related": [is_age_derived_structured_feature(column) for column in candidate_columns],
        "p4205_derived": [is_p4205_derived_structured_feature(column) for column in candidate_columns],
    })
    policy_table.to_csv(
        output_dir / "structured_feature_policy.csv", index=False, encoding="utf-8-sig")
    feature_meta["structured_dropped_columns"] = policy.dropped_features
    if args.structured_feature_set == "clinical_no_p4205_no_age":
        remaining_age = sorted(set(policy.kept_features).intersection(
            {"p21022", "age_squared", "smoking_exposure_x_age"}))
        if remaining_age:
            raise AssertionError(f"结构化特征策略审计发现年龄泄漏：{remaining_age}")
    waveform_config = WaveformConfig(
        expected_length=args.expected_length,
        normalize=not args.no_normalize,
        return_channels=args.wave_channels > 1,
        channel_count=args.wave_channels,
    )
    train_dataset = SAGMDataset(
        frame[EID_COLUMN].iloc[train_idx].to_numpy(),
        frame[WAVE_COLUMN].iloc[train_idx].tolist(),
        matrices["train_p4205"],
        matrices["train_p4205_cross"],
        matrices["train_structured"],
        labels[train_idx],
        masks[train_idx],
        waveform_config,
    )
    val_dataset = SAGMDataset(
        frame[EID_COLUMN].iloc[val_idx].to_numpy(),
        frame[WAVE_COLUMN].iloc[val_idx].tolist(),
        matrices["val_p4205"],
        matrices["val_p4205_cross"],
        matrices["val_structured"],
        labels[val_idx],
        masks[val_idx],
        waveform_config,
    )
    test_dataset = SAGMDataset(
        frame[EID_COLUMN].iloc[test_idx].to_numpy(),
        frame[WAVE_COLUMN].iloc[test_idx].tolist(),
        matrices["test_p4205"],
        matrices["test_p4205_cross"],
        matrices["test_structured"],
        labels[test_idx],
        masks[test_idx],
        waveform_config,
    )
    task_weights = resolve_task_weights(
        args.task_mode, args.task_weights, len(tasks))
    criterion = build_loss(
        args.loss,
        labels[train_idx],
        masks[train_idx],
        task_weights,
        device,
        args.focal_alpha,
        args.focal_gamma,
    )
    config = {
        "model_family": "SAGM-Net",
        "task_mode": args.task_mode,
        "tasks": tasks,
        "task_labels": {task: task_label_map(args.task_mode)[task] for task in tasks},
        "cohort": args.cohort,
        "target_years": args.target_years,
        "followup_dates": str(args.followup_dates.resolve()) if args.followup_dates is not None else None,
        "landmark_metadata": landmark_bundle.metadata if landmark_bundle is not None else None,
        "structured_feature_set": args.structured_feature_set,
        "structured_encoder": args.structured_encoder,
        "p4205_feature_encoder": args.p4205_feature_encoder,
        "vsn_variable_dim": int(args.vsn_variable_dim),
        "p4205_cross_mode": args.p4205_cross_mode,
        "data": str(args.data.resolve()),
        "p4205_cache": str(args.p4205_cache.resolve()),
        "effective_p4205_input": (
            effective_p4205_input(args.models[0])
            if len({effective_p4205_input(model_name) for model_name in args.models}) == 1
            else "mixed_by_model"
        ),
        "effective_p4205_input_by_model": {
            model_name: effective_p4205_input(model_name) for model_name in args.models
        },
        "row_count": int(len(frame)),
        "task_label_counts": {task: int(labels[masks[:, index] > 0, index].sum()) for index, task in enumerate(tasks)},
        "task_mask_counts": {task: int((masks[:, index] > 0).sum()) for index, task in enumerate(tasks)},
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "seed": int(args.seed),
        "train_seed": train_seed,
        "split_source": split_source,
        "device": str(device),
        "wave_channels": int(args.wave_channels),
        "expected_length": int(args.expected_length),
        "waveform_cache_policy": "cpu_float32_lazy_per_split_original_parser",
        "unused_wave_policy": "empty_placeholder_without_gpu_transfer",
        "wave_normalize": bool(not args.no_normalize),
        "inception_kernel_sets": args.inception_kernel_sets,
        "inception_stage_channels": args.inception_stage_channels,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "early_stop_metric": args.early_stop_metric,
        "early_stop_aggregation": early_stop_aggregation,
        "threshold_policy": "validation_selected_threshold_applied_to_test",
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "embed_dim": int(args.embed_dim),
        "loss": args.loss,
        "task_weights": task_weights[: len(tasks)].tolist(),
        "feature_meta": feature_meta,
        "checkpoint_enabled": bool(args.save_checkpoints),
        "permutation_repeats": int(args.permutation_repeats),
        "permutation_max_samples": int(args.permutation_max_samples),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config,
                                                           ensure_ascii=False, indent=2), encoding="utf-8")
    write_model_audit_note(output_dir, args, config, split_summary)

    result_payloads: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []
    test_prediction_lookup: dict[str, pd.DataFrame] = {}
    for ablation_mode in args.models:
        selected_encoders = args.wave_encoders if ablation_mode in WAVE_ENCODER_DEPENDENT_MODES else [
            "none"]
        for encoder in selected_encoders:
            actual_encoder = args.wave_encoders[0] if encoder == "none" else encoder
            model_name = f"{ablation_mode}_{encoder}" if encoder != "none" else ablation_mode
            model_dir = output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_seed = stable_model_seed(train_seed, model_name)
            set_global_seed(model_seed)
            for dataset in (train_dataset, val_dataset, test_dataset):
                dataset.configure_wave(
                    ablation_mode in WAVE_ENCODER_DEPENDENT_MODES)
            train_loader = make_loader(
                train_dataset, args.batch_size, True, args.num_workers, model_seed)
            val_loader = make_loader(
                val_dataset, args.batch_size, False, args.num_workers, model_seed + 1)
            test_loader = make_loader(
                test_dataset, args.batch_size, False, args.num_workers, model_seed + 2)
            print(
                f"[SAGM] start model={model_name} device={device} seed={model_seed}", flush=True)
            p4205_feature_dim = int(
                matrices["train_p4205_cross"].shape[1]
                if ablation_mode in P4205_CROSS_ABLATION_MODES
                else matrices["train_p4205"].shape[1]
            )
            model = SAGMNet(
                structured_dim=int(matrices["train_structured"].shape[1]),
                p4205_feature_dim=p4205_feature_dim,
                structured_encoder=args.structured_encoder,
                p4205_feature_encoder=args.p4205_feature_encoder,
                structured_group_indices=matrices["structured_vsn_group_indices"],
                p4205_group_indices=(
                    matrices["p4205_cross_vsn_group_indices"]
                    if ablation_mode in P4205_CROSS_ABLATION_MODES
                    else matrices["p4205_vsn_group_indices"]
                ),
                vsn_variable_dim=args.vsn_variable_dim,
                wave_channels=args.wave_channels,
                output_dim=len(tasks),
                wave_encoder=actual_encoder,
                ablation_mode=ablation_mode,
                embed_dim=args.embed_dim,
                dropout=args.dropout,
                inception_kernel_sets=args.inception_kernel_sets,
                inception_stage_channels=args.inception_stage_channels,
            ).to(device)
            payload, val_pred, test_pred, history = train_model(
                model, train_loader, val_loader, test_loader, criterion, args, device, tasks, early_stop_aggregation
            )
            payload["model_name"] = model_name
            payload["ablation_mode"] = ablation_mode
            payload["wave_encoder"] = encoder
            payload["model_seed"] = int(model_seed)
            # 参数量与最佳 epoch 共同保存，便于调参汇总时排除“仅因规模变大”的假增益。
            payload["parameter_count"] = int(
                sum(parameter.numel() for parameter in model.parameters()))
            payload["trainable_parameter_count"] = int(
                sum(p.numel() for p in model.parameters() if p.requires_grad))
            (model_dir / 'architecture.txt').write_text(str(model), encoding='utf-8')
            result_payloads[model_name] = payload
            history.to_csv(model_dir / "loss_history.csv",
                           index=False, encoding="utf-8-sig")
            if getattr(args, "cooperation_validation_only", False):
                torch.save({"state_dict": model.state_dict(), "config": config, "best_epoch": payload["best_epoch"]}, model_dir / "best_model.pt")
                saved = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
                model.load_state_dict(saved["state_dict"])
                check = collect_predictions(model, val_loader, criterion, device)
                np.testing.assert_allclose(check["probability"], val_pred["probability"], rtol=0, atol=1e-7)
                payload["checkpoint_validation_reload_passed"] = True
                save_prediction_table(model_dir / "validation_predictions.csv", frame[EID_COLUMN].iloc[val_idx].to_numpy(), val_pred, tasks)
                (model_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                (output_dir / "run_results.json").write_text(json.dumps(result_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
                print("主三塔验证筛查完成；最佳检查点已保存", flush=True)
                return
            if args.save_checkpoints:
                checkpoint_path = model_dir / "best_model.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_name": model_name,
                        "tasks": tasks,
                        "feature_meta": feature_meta,
                        "model_seed": int(model_seed),
                    },
                    checkpoint_path,
                )
                # 重载刚写入的最佳 checkpoint；若预测不一致则终止，避免留下不可复现实验。
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["state_dict"])
                reloaded_validation = collect_predictions(
                    model, val_loader, criterion, device)
                reloaded_test = collect_predictions(
                    model, test_loader, criterion, device)
                reload_differences: dict[str, float] = {}
                for split_name, original, reloaded in (
                    ("validation", val_pred, reloaded_validation),
                    ("test", test_pred, reloaded_test),
                ):
                    for field in ("probability", "gate_wave", "gate_feature"):
                        original_value = np.asarray(
                            original[field], dtype=np.float64)
                        reloaded_value = np.asarray(
                            reloaded[field], dtype=np.float64)
                        max_abs_diff = float(
                            np.max(np.abs(reloaded_value - original_value)))
                        reload_differences[f"{split_name}_{field}"] = max_abs_diff
                        if not np.allclose(reloaded_value, original_value, rtol=0.0, atol=1e-7):
                            raise AssertionError(
                                f"checkpoint 重载 {split_name}/{field} 不一致：max_abs_diff={max_abs_diff}"
                            )
                payload["checkpoint_reload_max_abs_diff"] = max(
                    reload_differences.values())
                payload["checkpoint_reload_audit"] = reload_differences
            save_prediction_table(
                model_dir / "validation_predictions.csv",
                frame[EID_COLUMN].iloc[val_idx].to_numpy(),
                val_pred,
                tasks,
            )
            save_prediction_table(
                model_dir / "test_predictions.csv",
                frame[EID_COLUMN].iloc[test_idx].to_numpy(),
                test_pred,
                tasks,
            )
            save_vsn_weight_artifacts(
                model_dir, frame[EID_COLUMN].iloc[test_idx].to_numpy(), test_pred, model)
            save_vector_gate_artifacts(
                model_dir, frame[EID_COLUMN].iloc[test_idx].to_numpy(), test_pred)
            (model_dir / "metrics.json").write_text(json.dumps(payload,
                                                               ensure_ascii=False, indent=2), encoding="utf-8")
            summary_rows.extend(flatten_metrics(
                model_name, encoder, payload, "validation"))
            summary_rows.extend(flatten_metrics(
                model_name, encoder, payload, "test"))
            test_prediction_lookup[model_name] = pd.read_csv(
                model_dir / "test_predictions.csv")
            if args.permutation_repeats > 0 and ablation_mode == "sagm_gated":
                permutation_detail = grouped_permutation_importance(
                    model=model,
                    wave_values=test_dataset.wave_values,
                    p4205=matrices["test_p4205"],
                    structured=matrices["test_structured"],
                    labels=labels[test_idx],
                    masks=masks[test_idx],
                    tasks=tasks,
                    waveform_config=waveform_config,
                    structured_column_indices=matrices["structured_column_indices"],
                    p4205_feature_groups=matrices["p4205_vsn_group_indices"],
                    seed=int(model_seed),
                    repeats=int(args.permutation_repeats),
                    batch_size=int(args.batch_size),
                    device=device,
                    max_samples=int(args.permutation_max_samples),
                )
                permutation_detail.to_csv(
                    model_dir / "permutation_importance_detail.csv", index=False, encoding="utf-8-sig")
                summarize_permutation_importance(permutation_detail).to_csv(
                    model_dir / "permutation_importance_summary.csv", index=False, encoding="utf-8-sig"
                )

    metrics = pd.DataFrame(summary_rows)
    metrics.to_csv(output_dir / "ablation_summary.csv",
                   index=False, encoding="utf-8-sig")
    primary_task = primary_task_name(tasks, early_stop_aggregation)
    test_primary = metrics[(metrics["split"].eq("test")) & (
        metrics["task"].eq(primary_task))].copy()
    paper = test_primary.sort_values(["pr_auc", "roc_auc"], ascending=False)
    paper.to_csv(output_dir / "paper_main_table.csv",
                 index=False, encoding="utf-8-sig")
    (output_dir / "run_results.json").write_text(
        json.dumps(result_payloads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    full_gated_names = {
        f"sagm_gated_{encoder}" for encoder in args.wave_encoders}
    best_sagm = paper[paper["model_name"].isin(full_gated_names)].head(1)
    if "all_cause" in tasks and not best_sagm.empty:
        best_name = str(best_sagm.iloc[0]["model_name"])
        gate_interpretability(frame.iloc[test_idx].copy(
        ), test_prediction_lookup[best_name], output_dir)
    baseline_name = "structured_only" if "structured_only" in test_prediction_lookup else "structured_only_resnet1d"
    if not args.disable_bootstrap and baseline_name in test_prediction_lookup and not best_sagm.empty:
        challenger_name = str(best_sagm.iloc[0]["model_name"])
        baseline = test_prediction_lookup[baseline_name]
        challenger = test_prediction_lookup[challenger_name]
        task_ci = [
            bootstrap_prediction_pair_ci(
                baseline,
                challenger,
                comparison=f"{challenger_name}_vs_{baseline_name}",
                repeats=args.bootstrap_repeats,
                seed=args.seed,
                task=task,
            )
            for task in tasks
        ]
        task_ci = [item for item in task_ci if not item.empty]
        if task_ci:
            pd.concat(task_ci, axis=0, ignore_index=True).to_csv(
                output_dir / "bootstrap_ci.csv", index=False, encoding="utf-8-sig"
            )
        tower_ci = build_tower_bootstrap_ci(
            test_prediction_lookup,
            args.wave_encoders,
            args.bootstrap_repeats,
            args.seed,
            tasks=tasks,
        )
        if not tower_ci.empty:
            tower_ci.to_csv(output_dir / "tower_bootstrap_ci.csv",
                            index=False, encoding="utf-8-sig")

    print(f"SAGM-Net 训练与评估完成：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
