"""
文件说明：读取 UK Biobank 随访日期并构建 landmark 时间窗所需的审计字段。

本模块只处理基线日期、死亡日期和行政截尾日期，绝不返回可用于模型输入的日期特征。
它为 0-5 年和 5-10 年 landmark 标签提供统一、可测试的风险集定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EID_COLUMN = "eid"
BASELINE_DATE_COLUMN = "p53_i0"
DEATH_DATE_COLUMN = "p40000_i0"
METADATA_COLUMNS = ("ukb_release_version", "extraction_date", "common_administrative_censor_date")
YEAR_DAYS = 365.25


@dataclass(frozen=True)
class FollowupBundle:
    """已验证的随访日期和元数据，避免调用方忽略行政截尾日期。"""

    dates: pd.DataFrame
    censor_date: pd.Timestamp
    metadata: dict[str, str]


def _parse_date(series: pd.Series, column: str) -> pd.Series:
    """严格解析日期，非空但无法解析的值会在调用处被审计并拒绝。"""
    parsed = pd.to_datetime(series, errors="coerce")
    invalid = series.notna() & parsed.isna()
    if invalid.any():
        sample = series.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{column} 含无法解析的日期，示例：{sample}")
    return parsed


def load_followup_workbook(path: Path) -> FollowupBundle:
    """读取固定双工作表格式的随访 Excel，并验证 EID 和截尾日期。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到随访日期工作簿：{path}")
    workbook = pd.ExcelFile(path)
    required_sheets = {"followup", "metadata"}
    missing_sheets = required_sheets.difference(workbook.sheet_names)
    if missing_sheets:
        raise ValueError(f"随访工作簿缺少工作表：{sorted(missing_sheets)}")
    dates = pd.read_excel(workbook, sheet_name="followup")
    missing_columns = {EID_COLUMN, BASELINE_DATE_COLUMN, DEATH_DATE_COLUMN}.difference(dates.columns)
    if missing_columns:
        raise ValueError(f"followup 缺少必需列：{sorted(missing_columns)}")
    dates = dates[[EID_COLUMN, BASELINE_DATE_COLUMN, DEATH_DATE_COLUMN]].copy()
    if dates[EID_COLUMN].isna().any() or dates[EID_COLUMN].duplicated().any():
        raise ValueError("followup 的 eid 必须非空且唯一。")
    dates[EID_COLUMN] = dates[EID_COLUMN].astype("string")
    dates[BASELINE_DATE_COLUMN] = _parse_date(dates[BASELINE_DATE_COLUMN], BASELINE_DATE_COLUMN)
    dates[DEATH_DATE_COLUMN] = _parse_date(dates[DEATH_DATE_COLUMN], DEATH_DATE_COLUMN)

    metadata_frame = pd.read_excel(workbook, sheet_name="metadata")
    missing_metadata = set(METADATA_COLUMNS).difference(metadata_frame.columns)
    if missing_metadata or len(metadata_frame) != 1:
        raise ValueError(
            "metadata 必须恰好一行并包含：" + ", ".join(METADATA_COLUMNS)
        )
    metadata = {column: str(metadata_frame.iloc[0][column]) for column in METADATA_COLUMNS}
    censor_date = pd.to_datetime(metadata["common_administrative_censor_date"], errors="coerce")
    if pd.isna(censor_date):
        raise ValueError("metadata.common_administrative_censor_date 不是有效日期。")
    return FollowupBundle(dates=dates, censor_date=pd.Timestamp(censor_date), metadata=metadata)


def attach_landmark_followup(frame: pd.DataFrame, bundle: FollowupBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 EID 合并随访日期并返回合格队列与可审计的排除统计。"""
    if EID_COLUMN not in frame:
        raise ValueError("模型数据缺少 eid，无法合并随访日期。")
    base = frame.copy()
    if base[EID_COLUMN].isna().any() or base[EID_COLUMN].duplicated().any():
        raise ValueError("模型数据的 eid 必须非空且唯一。")
    base[EID_COLUMN] = base[EID_COLUMN].astype("string")
    merged = base.merge(bundle.dates, on=EID_COLUMN, how="left", validate="one_to_one", indicator=True)
    baseline = merged[BASELINE_DATE_COLUMN]
    death = merged[DEATH_DATE_COLUMN]
    censor = bundle.censor_date
    years_to_censor = (censor - baseline).dt.total_seconds() / (86400.0 * YEAR_DAYS)
    years_to_death = (death - baseline).dt.total_seconds() / (86400.0 * YEAR_DAYS)

    reason = pd.Series("eligible", index=merged.index, dtype="string")
    reason.loc[merged["_merge"].ne("both")] = "missing_followup_eid"
    reason.loc[reason.eq("eligible") & baseline.isna()] = "missing_baseline_date"
    reason.loc[reason.eq("eligible") & death.notna() & death.le(baseline)] = "death_not_after_baseline"
    reason.loc[reason.eq("eligible") & death.notna() & death.gt(censor)] = "death_after_administrative_censor"
    # 未死亡者必须具有完整十年可观察期；已死亡者的事件时间本身已被观测到。
    reason.loc[reason.eq("eligible") & death.isna() & years_to_censor.lt(10.0)] = "insufficient_10y_followup"

    merged["landmark_years_to_death"] = years_to_death
    merged["landmark_years_to_censor"] = years_to_censor
    merged["landmark_censor_date"] = censor
    merged["landmark_exclusion_reason"] = reason
    audit = (
        reason.value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="count")
        .sort_values(["reason"], kind="stable")
        .reset_index(drop=True)
    )
    eligible = merged.loc[reason.eq("eligible")].drop(columns=["_merge"]).reset_index(drop=True)
    if eligible.empty:
        raise ValueError("日期审计后没有可用于 landmark 分析的样本。")
    return eligible, audit


def landmark_window_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """生成两个预先固定窗口的事件与风险集；晚期窗口严格排除早死者。"""
    if "landmark_years_to_death" not in frame:
        raise ValueError("缺少 landmark_years_to_death；请先调用 attach_landmark_followup。")
    time = pd.to_numeric(frame["landmark_years_to_death"], errors="coerce")
    died = time.notna()
    early_event = died & time.gt(0.0) & time.le(5.0)
    late_at_risk = ~early_event
    late_event = died & time.gt(5.0) & time.le(10.0)
    return {
        "all_cause_0_5": early_event,
        "all_cause_5_10": late_event,
        "late_at_risk": late_at_risk,
    }
