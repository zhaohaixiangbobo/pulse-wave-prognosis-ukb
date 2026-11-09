"""三塔工程分支的预注册候选：仅在训练分区验证清理依据、拟合列选择。"""
import numpy as np
import pandas as pd

CONSTANTS = {
    'p4205_signal_length': 100, 'p4205_declared_points': 100, 'p4205_parse_ratio': 1,
    'p4205_mean': 0, 'p4205_std': 1, 'p4205_energy': 100,
    'p4205_q_parse_success': 1, 'p4205_q_expected_length': 1, 'p4205_q_nonflat': 1,
    'p4205_q_h1_success': 1, 'p4205_q_width_success': 1, 'p4205_freq_peak1_rel_amplitude': 1,
}
ALIASES = {
    'p4205_skew': ('p4205_stat_norm_skew', 1),
    'p4205_kurtosis': ('p4205_stat_norm_kurtosis', 1),
    'p4205_dominant_freq': ('p4205_freq_dominant_freq_norm', 100),
    'p4205_time_width_period_ratio': ('p4205_time_width_one_third_ratio', 1),
    'p4205_raw_point_mean': ('p4205_stat_raw_mean', 1),
    'p4205_raw_point_std': ('p4205_stat_raw_std', 1),
    'p4205_raw_point_range': ('p4205_stat_raw_range', 1),
}
ANCHORS = ['p4205_stat_raw_mean', 'p4205_stat_raw_std',
           'p4205_stat_raw_median', 'p4205_stat_raw_iqr']
RECIPES = ('E0', 'E1', 'E2', 'E3')


def candidate_columns(columns, recipe):
    """候选定义固定，不按结局排名，不按相关系数自动删除。"""
    if recipe not in RECIPES:
        raise ValueError(f'未知候选：{recipe}')
    if recipe == 'E0':
        return list(columns)
    cleaned = [c for c in columns if c not in CONSTANTS and c not in ALIASES]
    if recipe == 'E1':
        return cleaned
    prefixes = ('p4205_time_', 'p4205_morph_', 'p4205_q_')
    if recipe == 'E3':
        prefixes += ('p4205_freq_', 'p4205_nonlinear_')
    return [c for c in cleaned if c.startswith(prefixes) or c in ANCHORS]


def verify_cleanup(train):
    """理论常量允许浮点舍入；缺失模式不一致不能当重复列删除。"""
    checks = []
    for name, value in CONSTANTS.items():
        x = train[name].to_numpy(float)
        passed = bool(np.isfinite(x).all() and np.allclose(
            x, value, rtol=1e-5, atol=1e-6))
        checks.append(
            {'feature': name, 'kind': 'theoretical_constant', 'passed': passed})
    for name, (other, scale) in ALIASES.items():
        x, y = train[name].to_numpy(
            float), train[other].to_numpy(float) * scale
        passed = bool(np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(
            x, y, rtol=1e-5, atol=1e-6, equal_nan=True))
        checks.append({'feature': name, 'kind': 'alias',
                      'representative': other, 'passed': passed})
    failed = [c['feature'] for c in checks if not c['passed']]
    if failed:
        raise ValueError(f'既有107项清单的训练数据核验失败，禁止静默放宽规则：{failed}')
    return checks


def fit_selection(frame, train_idx, recipe):
    """返回冻结列清单及审计；不读取验证或测试分区的统计量。"""
    train = frame.iloc[train_idx].replace([np.inf, -np.inf], np.nan)
    nominal = candidate_columns(frame.columns, recipe)
    checks = verify_cleanup(train) if recipe != 'E0' else []
    kept, removed = [], []
    for column in nominal:
        if recipe != 'E0':
            # 观测值相同但缺失模式不同仍可能携带信息，不当作严格常量删除。
            if train[column].isna().all() or train[column].nunique(dropna=False) == 1:
                removed.append(
                    {'feature': column, 'reason': 'training_empty_or_constant'})
                continue
            duplicate = next(
                (c for c in kept if train[c].equals(train[column])), None)
            if duplicate is not None:
                removed.append(
                    {'feature': column, 'reason': 'training_exact_duplicate', 'representative': duplicate})
                continue
        kept.append(column)
    if not kept:
        raise ValueError('工程分支没有有效输入。')
    quality = pd.DataFrame({'feature': train.columns,
                            'missing_fraction': train.isna().mean().to_numpy(),
                            'unique_nonmissing': train.nunique(dropna=True).to_numpy(),
                            'most_common_fraction_nonmissing': [float(train[c].value_counts(normalize=True).max())
                                                                if train[c].notna().any() else None for c in train]})
    manifest = {'recipe': recipe, 'original_count': len(frame.columns), 'nominal_count': len(nominal),
                'selected_count': len(kept), 'train_rows': len(train), 'selected': kept,
                'excluded_by_definition': [c for c in frame if c not in nominal],
                'removed': removed, 'cleanup_checks': checks,
                'quality': quality.astype(object).where(pd.notna(quality), None).to_dict('records'),
                'fit_scope': 'training_only_no_outcomes'}
    return kept, manifest
