"""
文件说明：统一管理 P4205 派生结构化特征的识别与剔除策略。

本模块用于构建严格的 clinical-only / clinical_no_p4205 基线，避免 `p21021_i0`、
`p4199_i0`、`p4200_i0` 及其派生交叉项提前吸收 P4205 波形信号，从而压缩 raw
P4205 分支的增量评估。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


STRUCTURED_FEATURE_SET_CHOICES = ("current", "clinical_no_p4205", "clinical_no_p4205_no_age")

AGE_DERIVED_STRUCTURED_FEATURES = {
    "p21022",
    "age_squared",
    "smoking_exposure_x_age",
}

P4205_DERIVED_STRUCTURED_FEATURES = {
    "p21021_i0",
    "p4199_i0",
    "p4200_i0",
    "stiffness_x_age",
    "stiffness_per_age",
    "notch_shoulder_gap",
    "notch_shoulder_midpoint",
    "notch_position_ratio",
    "shoulder_position_ratio",
    "baseline_missing_count",
}

P4205_DERIVED_REASON = {
    "p21021_i0": "UKB pulse wave arterial stiffness index，直接来自 P4205 相关测量。",
    "p4199_i0": "pulse wave notch position，属于 P4205 波形形态摘要。",
    "p4200_i0": "pulse wave shoulder position，属于 P4205 波形形态摘要。",
    "stiffness_x_age": "由 p21021_i0 与年龄相乘得到，仍携带 P4205 派生信号。",
    "stiffness_per_age": "由 p21021_i0 与年龄构造，仍携带 P4205 派生信号。",
    "notch_shoulder_gap": "由 p4199_i0 与 p4200_i0 构造，仍携带 P4205 派生信号。",
    "notch_shoulder_midpoint": "由 p4199_i0 与 p4200_i0 构造，仍携带 P4205 派生信号。",
    "notch_position_ratio": "由 p4199_i0 归一化得到，仍携带 P4205 派生信号。",
    "shoulder_position_ratio": "由 p4200_i0 归一化得到，仍携带 P4205 派生信号。",
    "baseline_missing_count": "当前缺失计数包含 P4205 派生结构化字段，严格基线中先剔除。",
    "p21022": "基线年龄；去年龄消融中仅从模型输入删除，仍保留用于队列筛选和结局构建。",
    "age_squared": "由基线年龄构造；去年龄消融中删除。",
    "smoking_exposure_x_age": "含年龄的交叉项；去年龄消融中删除。",
}


@dataclass(frozen=True)
class FeaturePolicyResult:
    """记录结构化特征策略执行结果，便于写入配置和审计表。"""

    feature_set: str
    kept_features: list[str]
    dropped_features: list[str]


def validate_structured_feature_set(feature_set: str) -> None:
    """检查结构化特征口径名称，尽早暴露命令行拼写错误。"""
    if feature_set not in STRUCTURED_FEATURE_SET_CHOICES:
        raise ValueError(f"未知结构化特征口径：{feature_set}，可选 {STRUCTURED_FEATURE_SET_CHOICES}。")


def is_age_derived_structured_feature(feature: str) -> bool:
    """判断结构化列是否直接使用年龄，供严格去年龄消融使用。"""
    return str(feature) in AGE_DERIVED_STRUCTURED_FEATURES


def is_p4205_derived_structured_feature(feature: str) -> bool:
    """判断结构化列是否直接或间接来自 P4205 波形摘要。"""
    return str(feature) in P4205_DERIVED_STRUCTURED_FEATURES


def split_structured_features(columns: list[str] | pd.Index, feature_set: str) -> FeaturePolicyResult:
    """按指定口径拆分保留列与剔除列。"""
    validate_structured_feature_set(feature_set)
    names = [str(column) for column in columns]
    if feature_set == "current":
        return FeaturePolicyResult(feature_set=feature_set, kept_features=names, dropped_features=[])
    dropped_set = {name for name in names if is_p4205_derived_structured_feature(name)}
    if feature_set == "clinical_no_p4205_no_age":
        dropped_set.update(name for name in names if is_age_derived_structured_feature(name))
    dropped = [name for name in names if name in dropped_set]
    kept = [name for name in names if name not in dropped_set]
    return FeaturePolicyResult(feature_set=feature_set, kept_features=kept, dropped_features=dropped)


def drop_p4205_derived_structured_features(
    frame: pd.DataFrame, feature_set: str
) -> tuple[pd.DataFrame, FeaturePolicyResult]:
    """返回指定口径的结构化特征表，并给出剔除明细。"""
    policy = split_structured_features(frame.columns, feature_set)
    return frame.loc[:, policy.kept_features].copy(), policy


def feature_policy_table(columns: list[str] | pd.Index) -> pd.DataFrame:
    """生成审计表，说明 current 与 clinical_no_p4205 的列级差异。"""
    rows = []
    for name in [str(column) for column in columns]:
        is_derived = is_p4205_derived_structured_feature(name)
        rows.append(
            {
                "feature": name,
                "keep_in_current": True,
                "keep_in_clinical_no_p4205": not is_derived,
                "p4205_derived": is_derived,
                "reason": P4205_DERIVED_REASON.get(name, "非 P4205 派生结构化特征。"),
            }
        )
    return pd.DataFrame(rows)
