"""统一精确日期标签：仅由基线/死亡日期及行政截止日生成，供数据构建和边界测试复用。"""
import pandas as pd


def derive_labels(frame, cutoff):
    """保留已完成十年观察的截止日后死亡者；排除无法确认十年结局的记录。"""
    b = pd.to_datetime(frame['p53_i0'], errors='coerce')
    d = pd.to_datetime(frame['p40000_i0'], errors='coerce')
    c = pd.Timestamp(cutoff)
    observed = d.notna() & d.le(c)
    duration = (d - b).dt.total_seconds() / (86400 * 365.25)
    follow = (c - b).dt.total_seconds() / (86400 * 365.25)
    reason = pd.Series('eligible', index=frame.index)
    reason.loc[b.isna()] = 'missing_baseline_date'
    invalid = frame['p40000_i0'].notna() & d.isna()
    reason.loc[reason.eq('eligible') & invalid] = 'invalid_death_date'
    reason.loc[reason.eq('eligible') & b.gt(c)] = 'baseline_after_cutoff'
    reason.loc[reason.eq('eligible') & d.notna() & d.le(b)] = 'death_not_after_baseline'
    # 死亡年龄仅用于发现已知死亡却缺少日期的记录，不参与日期标签计算。
    if 'p40007_i0' in frame:
        reason.loc[reason.eq('eligible') & frame.p40007_i0.notna() & d.isna()] = 'known_death_missing_date'
    reason.loc[reason.eq('eligible') & ~observed & follow.lt(10)] = 'insufficient_10y_followup'
    t = duration.where(observed)
    return pd.DataFrame({'exact_target': (t.gt(0) & t.le(10)).astype('int8'),
                         'landmark_years_to_death': t,
                         'exact_early': (t.gt(0) & t.le(5)).astype('int8'),
                         'exact_late': (t.gt(5) & t.le(10)).astype('int8'),
                         'exact_late_mask': (~(t.gt(0) & t.le(5))).astype('int8'),
                         'potential_years': follow, 'exclusion_reason': reason})
