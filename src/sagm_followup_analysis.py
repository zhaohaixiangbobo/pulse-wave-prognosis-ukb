"""
文件说明：年龄亚组样本分布、预测效果和分组置换归因的通用工具。

归因采用测试时分组置换，量化输入组被打乱后模型判别指标的下降；该结果反映模型
关联性依赖，不作因果解释，也不要求额外安装 SHAP 或 Captum。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from train_two_tower_classification import AGE_COLUMN, EID_COLUMN
from two_tower_torch_model import WaveformConfig, parse_p4205_i0_to_tensor


AGE_GROUP_SPECS = (
    ("primary", "39-59", 39, 59),
    ("primary", "60-72", 60, 72),
    ("fine", "39-49", 39, 49),
    ("fine", "50-54", 50, 54),
    ("fine", "55-59", 55, 59),
    ("fine", "60-64", 60, 64),
    ("fine", "65-72", 65, 72),
)


def age_group_masks(frame: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """按预先指定的主分析和细分年龄段生成互斥掩码。"""
    age = pd.to_numeric(frame[AGE_COLUMN], errors="coerce")
    return [
        (scheme, label, age.between(low, high, inclusive="both").to_numpy(dtype=bool))
        for scheme, label, low, high in AGE_GROUP_SPECS
    ]


def age_split_distribution(
    frame: pd.DataFrame,
    split_indices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    masks: np.ndarray,
    tasks: list[str],
) -> pd.DataFrame:
    """导出每个年龄组、split 和任务的可观察样本量、阳性数及阳性率。"""
    rows: list[dict[str, object]] = []
    for split, split_idx in split_indices.items():
        local = frame.iloc[split_idx].reset_index(drop=True)
        for scheme, age_group, age_mask in age_group_masks(local):
            for task_index, task in enumerate(tasks):
                observed = masks[split_idx, task_index] > 0
                eligible = age_mask & observed
                positive = int(labels[split_idx, task_index][eligible].sum())
                n = int(eligible.sum())
                rows.append({
                    "split": split,
                    "age_scheme": scheme,
                    "age_group": age_group,
                    "task": task,
                    "n": n,
                    "positive_count": positive,
                    "positive_rate": float(positive / n) if n else np.nan,
                    "auc_reportable": bool(positive >= 20 and positive < n),
                })
    return pd.DataFrame(rows)


def subgroup_prediction_metrics(
    frame: pd.DataFrame,
    prediction: pd.DataFrame,
    tasks: list[str],
    model_name: str,
    min_positive: int = 20,
) -> pd.DataFrame:
    """在固定测试 EID 上计算年龄亚组逐头指标，并保留样本不足标记。"""
    merged = frame[[EID_COLUMN, AGE_COLUMN]].merge(prediction, on=EID_COLUMN, how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for scheme, age_group, age_mask in age_group_masks(merged):
        for task in tasks:
            keep = age_mask & merged[f"{task}_mask"].to_numpy(dtype=bool)
            y = merged.loc[keep, f"{task}_true"].to_numpy(dtype=np.int8)
            p = merged.loc[keep, f"{task}_probability"].to_numpy(dtype=np.float64)
            n = int(len(y))
            positive = int(y.sum())
            reportable = positive >= min_positive and positive < n
            rows.append({
                "model_name": model_name,
                "age_scheme": scheme,
                "age_group": age_group,
                "task": task,
                "n": n,
                "positive_count": positive,
                "positive_rate": float(positive / n) if n else np.nan,
                "auc_reportable": bool(reportable),
                "roc_auc": float(roc_auc_score(y, p)) if reportable else np.nan,
                "pr_auc": float(average_precision_score(y, p)) if reportable else np.nan,
            })
    return pd.DataFrame(rows)


def _predict_probability(
    model: torch.nn.Module,
    wave: np.ndarray,
    p4205: np.ndarray,
    structured: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """对 Numpy 输入分批前向，避免置换分析一次占用过多 GPU 内存。"""
    result: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(wave), batch_size):
            end = min(len(wave), start + batch_size)
            logits = model(
                torch.as_tensor(wave[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(p4205[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(structured[start:end], dtype=torch.float32, device=device),
            )
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result, axis=0)


def _metric_rows(
    original: np.ndarray,
    permuted: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    tasks: list[str],
    group: str,
    repeat: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, task in enumerate(tasks):
        keep = masks[:, index] > 0
        y = labels[keep, index].astype(np.int8)
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue
        base = original[keep, index]
        changed = permuted[keep, index]
        rows.extend([
            {"group": group, "repeat": repeat, "task": task, "metric": "roc_auc", "baseline": float(roc_auc_score(y, base)), "permuted": float(roc_auc_score(y, changed))},
            {"group": group, "repeat": repeat, "task": task, "metric": "pr_auc", "baseline": float(average_precision_score(y, base)), "permuted": float(average_precision_score(y, changed))},
        ])
    for row in rows:
        row["importance_drop"] = float(row["baseline"] - row["permuted"])
    return rows


def _p4205_domain_groups(feature_groups: Mapping[str, list[int]]) -> dict[str, list[int]]:
    """按工程特征命名前缀归并六类生理/信号领域，缺失指示随原特征同组置换。"""
    domains = {
        "p4205_stat": "p4205_stat",
        "p4205_time": "p4205_time",
        "p4205_morph": "p4205_morph",
        "p4205_freq": "p4205_freq",
        "p4205_nonlinear": "p4205_nonlinear",
        "p4205_raw_diff": "p4205_raw_",
    }
    result: dict[str, list[int]] = {name: [] for name in domains}
    for feature, indices in feature_groups.items():
        assigned = False
        for name, prefix in domains.items():
            if feature.startswith(prefix) or (name == "p4205_raw_diff" and feature.startswith("p4205_diff_")):
                result[name].extend(indices)
                assigned = True
                break
        if not assigned:
            result.setdefault("p4205_other_engineered", []).extend(indices)
    return {name: sorted(set(indices)) for name, indices in result.items() if indices}


def grouped_permutation_importance(
    model: torch.nn.Module,
    wave_values: list[object],
    p4205: np.ndarray,
    structured: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    tasks: list[str],
    waveform_config: WaveformConfig,
    structured_column_indices: Mapping[str, list[int]],
    p4205_feature_groups: Mapping[str, list[int]],
    seed: int,
    repeats: int,
    batch_size: int,
    device: torch.device,
    max_samples: int = 0,
) -> pd.DataFrame:
    """执行结构化、工程特征领域及原始波形窗口的测试时分组置换归因。"""
    if repeats <= 0:
        return pd.DataFrame()
    sample_index = np.arange(len(wave_values))
    if max_samples > 0 and max_samples < len(sample_index):
        rng = np.random.default_rng(seed)
        sample_index = np.sort(rng.choice(sample_index, size=max_samples, replace=False))
    wave = np.stack([
        parse_p4205_i0_to_tensor(wave_values[int(i)], config=waveform_config).numpy()
        for i in sample_index
    ]).astype(np.float32, copy=False)
    p4205_local = p4205[sample_index].astype(np.float32, copy=False)
    structured_local = structured[sample_index].astype(np.float32, copy=False)
    labels_local = labels[sample_index]
    masks_local = masks[sample_index]
    baseline = _predict_probability(model, wave, p4205_local, structured_local, batch_size, device)
    groups: dict[str, tuple[str, object]] = {}
    age_indices = sorted(set(
        structured_column_indices.get("p21022", [])
        + structured_column_indices.get("age_squared", [])
        + structured_column_indices.get("smoking_exposure_x_age", [])
    ))
    if age_indices:
        groups["structured_age_bundle"] = ("structured", age_indices)
    for name, columns in {
        "structured_smoking": ["p20160_i0", "p20162_i0", "smoking_exposure_x_age"],
        "structured_symptoms": ["p3616_i0", "p3751_i0", "p4717_i0", "p2316_i0", "symptom_yes_count"],
        "structured_sex": ["sex"],
    }.items():
        indices = sorted(set(index for column in columns for index in structured_column_indices.get(column, [])))
        if indices:
            groups[name] = ("structured", indices)
    for name, indices in _p4205_domain_groups(p4205_feature_groups).items():
        groups[name] = ("p4205", indices)
    window_size = max(1, wave.shape[2] // 10)
    for channel in range(wave.shape[1]):
        for window in range(10):
            start = window * window_size
            end = wave.shape[2] if window == 9 else min(wave.shape[2], (window + 1) * window_size)
            groups[f"wave_ch{channel}_win{window}"] = ("wave", (channel, start, end))

    rows: list[dict[str, object]] = []
    for group_index, (group, (kind, spec)) in enumerate(groups.items()):
        for repeat in range(repeats):
            rng = np.random.default_rng(int(seed + 1009 * (group_index + 1) + repeat))
            permutation = rng.permutation(len(sample_index))
            changed_wave, changed_p4205, changed_structured = wave, p4205_local, structured_local
            if kind == "structured":
                changed_structured = structured_local.copy()
                changed_structured[:, spec] = structured_local[permutation][:, spec]
            elif kind == "p4205":
                changed_p4205 = p4205_local.copy()
                changed_p4205[:, spec] = p4205_local[permutation][:, spec]
            else:
                channel, start, end = spec
                changed_wave = wave.copy()
                changed_wave[:, channel, start:end] = wave[permutation, channel, start:end]
            changed = _predict_probability(model, changed_wave, changed_p4205, changed_structured, batch_size, device)
            rows.extend(_metric_rows(baseline, changed, labels_local, masks_local, tasks, group, repeat))
    return pd.DataFrame(rows)


def summarize_permutation_importance(detail: pd.DataFrame) -> pd.DataFrame:
    """按任务、输入组和指标汇总多次置换的重要性下降。"""
    if detail.empty:
        return detail
    return detail.groupby(["group", "task", "metric"], as_index=False).agg(
        importance_drop_mean=("importance_drop", "mean"),
        importance_drop_sd=("importance_drop", "std"),
        repeats=("repeat", "nunique"),
        baseline_mean=("baseline", "mean"),
    )
