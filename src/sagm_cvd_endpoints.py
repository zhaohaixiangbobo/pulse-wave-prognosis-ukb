"""
文件说明：为 SAGM-Net 提供心血管三口径的 ICD-10 标签和分层切分工具。

本模块只处理结局标签与切分层级，不参与特征构建或模型前向传播。这样可确保
`p40001_i0` 和 `p41270` 永远不会意外进入模型输入，同时让标签边界可单测、可审计。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from landmark_followup import landmark_window_masks


CAUSE_COLUMN = "p40001_i0"
DIAGNOSIS_COLUMN = "p41270"

LEGACY_TASKS = ("all_cause", "circulatory_I", "respiratory_J", "malignant_neoplasm_C")
CVD_DEATH_TASKS = (
    "all_cause",
    "cv_wide_I00_I99",
    "cv_narrow_I00_I52",
    "coronary_I20_I25",
    "respiratory_J",
    "malignant_neoplasm_C",
)
RESPIRATORY_SUBTYPE_TASKS = (
    "resp_chronic_lower_airway_J40_J47",
    "resp_pneumonia_infection_J09_J22",
    "resp_interstitial_fibrotic_J60_J70_J80_J84",
)
CVD_DEATH_RESP_TASKS = (
    "all_cause", "cv_wide_I00_I99", "cv_narrow_I00_I52", "coronary_I20_I25",
    "respiratory_J", *RESPIRATORY_SUBTYPE_TASKS, "malignant_neoplasm_C",
)
CVD_DIAGNOSIS_TASKS = ("cv_narrow_I00_I52", "cv_wide_I00_I99", "coronary_I20_I25")
LANDMARK_DEATH_TASKS = (
    "all_cause_0_5",
    "cvd_wide_0_5",
    "all_cause_5_10",
    "cvd_wide_5_10",
)

TASK_LABELS = {
    "all_cause": "10年全因死亡",
    "circulatory_I": "10年循环系统死亡",
    "cv_wide_I00_I99": "10年宽口径心血管死亡（I00-I99）",
    "cv_narrow_I00_I52": "10年窄口径心脏疾病死亡（I00-I52）",
    "coronary_I20_I25": "10年冠心病死亡（I20-I25）",
    "respiratory_J": "10年呼吸系统死亡",
    "resp_chronic_lower_airway_J40_J47": "10年慢性下气道疾病死亡（J40-J47）",
    "resp_pneumonia_infection_J09_J22": "10年肺炎或急性感染死亡（J09-J22）",
    "resp_interstitial_fibrotic_J60_J70_J80_J84": "10年间质性或纤维化呼吸疾病死亡（J60-J70、J80-J84）",
    "malignant_neoplasm_C": "10年恶性肿瘤死亡",
    "all_cause_0_5": "基线后0-5年全因死亡",
    "cvd_wide_0_5": "基线后0-5年宽口径心血管死亡（I00-I99）",
    "all_cause_5_10": "存活至5年后的5-10年全因死亡",
    "cvd_wide_5_10": "存活至5年后的5-10年宽口径心血管死亡（I00-I99）",
}


DIAGNOSIS_TASK_LABELS = {
    "cv_narrow_I00_I52": "既往或累计窄口径心脏疾病诊断（I00-I52）",
    "cv_wide_I00_I99": "既往或累计宽口径心血管诊断（I00-I99）",
    "coronary_I20_I25": "既往或累计冠心病诊断（I20-I25）",
}


def task_label_map(task_mode: str) -> dict[str, str]:
    """返回与任务来源一致的中文标签，避免把 p41270 误写成10年死因。"""
    if task_mode == "cvd_diagnosis_mtl":
        return DIAGNOSIS_TASK_LABELS
    return TASK_LABELS

def task_names(task_mode: str) -> list[str]:
    """按实验模式返回固定且可审计的输出头顺序。"""
    mapping = {
        "binary": ("all_cause",),
        "mtl": LEGACY_TASKS,
        "cvd_death_mtl": CVD_DEATH_TASKS,
        "cvd_death_resp_mtl": CVD_DEATH_RESP_TASKS,
        "cvd_diagnosis_mtl": CVD_DIAGNOSIS_TASKS,
        "landmark_death_mtl": LANDMARK_DEATH_TASKS,
    }
    try:
        return list(mapping[task_mode])
    except KeyError as error:
        raise ValueError(f"未知 task_mode：{task_mode}") from error


def add_primary_cause_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """解析单个主要死因的 ICD-10 章节和两位数字，不修改原始字段。"""
    result = frame.copy()
    text = result[CAUSE_COLUMN].astype("string").str.strip()
    result["icd10_code"] = text.str.extract(r"^([A-Z][0-9]{2}(?:\.[0-9A-Z]+)?)", expand=False)
    result["icd10_chapter"] = result["icd10_code"].str[0]
    result["icd10_number"] = pd.to_numeric(
        result["icd10_code"].str.extract(r"^[A-Z]([0-9]{2})", expand=False), errors="coerce"
    )
    return result


def diagnosis_i_numbers(values: pd.Series) -> pd.Series:
    """从多码诊断字符串中提取全部 I00-I99 两位码，避免将 Z86.7 误判为 I 章疾病。"""
    pattern = re.compile(r"(?<![A-Z0-9])I([0-9]{2})(?:\.[0-9A-Z]+)?")
    return values.astype("string").fillna("").map(lambda value: tuple(int(item) for item in pattern.findall(value)))


def _cause_range(frame: pd.DataFrame, low: int, high: int) -> np.ndarray:
    return (
        frame["icd10_chapter"].eq("I")
        & frame["icd10_number"].between(low, high, inclusive="both")
    ).fillna(False).to_numpy(dtype=bool)


def _respiratory_range(frame: pd.DataFrame, *ranges: tuple[int, int]) -> np.ndarray:
    """判断主要死因是否属于指定的 J 章两位码区间。"""
    number = frame["icd10_number"]
    matched = pd.Series(False, index=frame.index)
    for low, high in ranges:
        matched |= number.between(low, high, inclusive="both")
    return (frame["icd10_chapter"].eq("J") & matched).fillna(False).to_numpy(dtype=bool)


def _diagnosis_range(numbers: pd.Series, low: int, high: int) -> np.ndarray:
    return numbers.map(lambda values: any(low <= value <= high for value in values)).to_numpy(dtype=bool)


def build_labels(frame: pd.DataFrame, task_mode: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """构建标签和 mask；未知死因死亡及缺失诊断记录不会被当成阴性。"""
    tasks = task_names(task_mode)
    labels = np.zeros((len(frame), len(tasks)), dtype=np.float32)
    masks = np.ones_like(labels, dtype=np.float32)
    target_positive = frame["target"].eq(1).to_numpy(dtype=bool)

    if task_mode == "landmark_death_mtl":
        windows = landmark_window_masks(frame)
        early_event = windows["all_cause_0_5"].to_numpy(dtype=bool)
        late_event = windows["all_cause_5_10"].to_numpy(dtype=bool)
        late_at_risk = windows["late_at_risk"].to_numpy(dtype=bool)
        wide = _cause_range(frame, 0, 99)
        unknown_early = early_event & frame["icd10_code"].isna().to_numpy(dtype=bool)
        unknown_late = late_event & frame["icd10_code"].isna().to_numpy(dtype=bool)
        values = {
            "all_cause_0_5": early_event,
            "cvd_wide_0_5": early_event & wide,
            "all_cause_5_10": late_event,
            "cvd_wide_5_10": late_event & wide,
        }
        task_masks = {
            "all_cause_0_5": np.ones(len(frame), dtype=bool),
            "cvd_wide_0_5": ~unknown_early,
            "all_cause_5_10": late_at_risk,
            "cvd_wide_5_10": late_at_risk & ~unknown_late,
        }
        for index, task in enumerate(tasks):
            labels[:, index] = values[task].astype(np.float32)
            masks[:, index] = task_masks[task].astype(np.float32)
        return tasks, labels, masks
    if task_mode == "cvd_diagnosis_mtl":
        observed = frame[DIAGNOSIS_COLUMN].notna().to_numpy(dtype=bool)
        numbers = diagnosis_i_numbers(frame[DIAGNOSIS_COLUMN])
        ranges = {
            "cv_narrow_I00_I52": _diagnosis_range(numbers, 0, 52),
            "cv_wide_I00_I99": _diagnosis_range(numbers, 0, 99),
            "coronary_I20_I25": _diagnosis_range(numbers, 20, 25),
        }
        for index, task in enumerate(tasks):
            labels[:, index] = ranges[task].astype(np.float32)
            masks[:, index] = observed.astype(np.float32)
        return tasks, labels, masks

    unknown_cause_death = frame["target"].eq(1) & frame["icd10_code"].isna()
    chapter = frame["icd10_chapter"].fillna("").astype(str)
    cause_masks = (~unknown_cause_death).to_numpy(dtype=np.float32)
    cause_ranges = {
        "cv_wide_I00_I99": _cause_range(frame, 0, 99),
        "cv_narrow_I00_I52": _cause_range(frame, 0, 52),
        "coronary_I20_I25": _cause_range(frame, 20, 25),
        "resp_chronic_lower_airway_J40_J47": _respiratory_range(frame, (40, 47)),
        "resp_pneumonia_infection_J09_J22": _respiratory_range(frame, (9, 22)),
        "resp_interstitial_fibrotic_J60_J70_J80_J84": _respiratory_range(frame, (60, 70), (80, 84)),
    }
    for index, task in enumerate(tasks):
        if task == "all_cause":
            labels[:, index] = target_positive.astype(np.float32)
            continue
        if task == "circulatory_I":
            local = chapter.eq("I").to_numpy(dtype=bool)
        elif task == "respiratory_J":
            local = chapter.eq("J").to_numpy(dtype=bool)
        elif task == "malignant_neoplasm_C":
            local = chapter.eq("C").to_numpy(dtype=bool)
        else:
            local = cause_ranges[task]
        labels[:, index] = (target_positive & local).astype(np.float32)
        masks[:, index] = cause_masks
    return tasks, labels, masks


def hierarchy_strata(tasks: list[str], labels: np.ndarray, masks: np.ndarray, task_mode: str) -> np.ndarray:
    """构建互斥层级 strata，保证嵌套心血管结局在三段切分中均衡。"""
    positions = {task: index for index, task in enumerate(tasks)}
    if task_mode == "cvd_diagnosis_mtl":
        observed = masks[:, 0] > 0
        wide = labels[:, positions["cv_wide_I00_I99"]] > 0
        narrow = labels[:, positions["cv_narrow_I00_I52"]] > 0
        coronary = labels[:, positions["coronary_I20_I25"]] > 0
        strata = np.full(len(labels), 1, dtype=np.int8)  # 有记录且无心血管诊断
        strata[~observed] = 0
        strata[wide & ~narrow] = 2
        strata[narrow & ~coronary] = 3
        strata[coronary] = 4
        return strata

    if task_mode == "landmark_death_mtl":
        positions = {task: index for index, task in enumerate(tasks)}
        early = labels[:, positions["all_cause_0_5"]] > 0
        late = labels[:, positions["all_cause_5_10"]] > 0
        early_wide = labels[:, positions["cvd_wide_0_5"]] > 0
        late_wide = labels[:, positions["cvd_wide_5_10"]] > 0
        strata = np.zeros(len(labels), dtype=np.int8)  # 十年内无死亡
        strata[early & ~early_wide] = 1
        strata[early_wide] = 2
        strata[late & ~late_wide] = 3
        strata[late_wide] = 4
        return strata
    if task_mode not in {"cvd_death_mtl", "cvd_death_resp_mtl"}:
        return labels[:, 0].astype(np.int8)
    all_cause = labels[:, positions["all_cause"]] > 0
    wide = labels[:, positions["cv_wide_I00_I99"]] > 0
    narrow = labels[:, positions["cv_narrow_I00_I52"]] > 0
    coronary = labels[:, positions["coronary_I20_I25"]] > 0
    known_cause = masks[:, positions["cv_wide_I00_I99"]] > 0
    strata = np.zeros(len(labels), dtype=np.int8)  # 无10年死亡
    strata[all_cause & ~known_cause] = 1
    strata[all_cause & known_cause & ~wide] = 2
    strata[wide & ~narrow] = 3
    strata[narrow & ~coronary] = 4
    strata[coronary] = 5
    if task_mode == "cvd_death_resp_mtl":
        chronic = labels[:, positions["resp_chronic_lower_airway_J40_J47"]] > 0
        infection = labels[:, positions["resp_pneumonia_infection_J09_J22"]] > 0
        interstitial = labels[:, positions["resp_interstitial_fibrotic_J60_J70_J80_J84"]] > 0
        broad_resp = labels[:, positions["respiratory_J"]] > 0
        strata[chronic] = 6
        strata[infection] = 7
        strata[interstitial] = 8
        strata[broad_resp & ~(chronic | infection | interstitial)] = 9
    return strata
