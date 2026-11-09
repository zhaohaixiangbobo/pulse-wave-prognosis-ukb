"""日期版训练数据入口：读取冻结目标并实施预先定义的结构化特征变体。"""
from pathlib import Path
import os
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[2]


def load_exact_data(path, max_rows):
    """拒绝缺少日期标签的数据，避免回退到旧年龄标签。"""
    from train_two_tower_classification import READ_COLUMNS
    extra = ['exact_target','landmark_years_to_death','exact_early','exact_late','exact_late_mask']
    df = pd.read_csv(path, usecols=READ_COLUMNS+extra, nrows=max_rows, low_memory=False)
    if not df.exact_target.isin([0,1]).all() or df.eid.duplicated().any():
        raise ValueError('日期标签或EID非法')
    df['target'] = df.exact_target.astype('int8')
    return df


def apply_variant(structured, source):
    """变体只改变训练输入；标签/日期字段始终禁止作为特征。"""
    variant = os.environ.get('REVISION_VARIANT','main')
    if variant == 'symptom_drop':
        structured = structured.drop(columns=['p3751_i0','p3616_i0','symptom_yes_count'], errors='raise')
    elif variant == 'clinical_extended':
        spec = json.loads((BASE/'configs/clinical_extension.json').read_text(encoding='utf-8'))
        if spec['status'] != 'verified' or not spec['columns']:
            raise ValueError('增强临床字段尚未核实，禁止运行该实验')
        clinical = pd.read_csv(BASE/'data/clinical_baseline.csv', dtype={'eid':str})
        if clinical.eid.duplicated().any():
            raise ValueError('临床补充表EID重复')
        deny = ['target','death','p400','p53','date','p412']
        columns = spec['columns']
        if any(any(x in c.lower() for x in deny) for c in columns):
            raise ValueError('临床扩展包含结局/日期/累计诊断字段')
        aligned = clinical.set_index('eid').reindex(source.eid.astype(str))
        for c in columns:
            if c in structured:
                raise ValueError('扩展字段与现有特征重复')
            structured[c] = aligned[c].to_numpy()
    elif variant != 'main':
        raise ValueError(f'未知变体{variant}')
    return structured
