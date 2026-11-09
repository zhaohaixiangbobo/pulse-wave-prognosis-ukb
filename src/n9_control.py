"""N9 后期全面加大正则：验证平台后降 lr、加大 weight decay 有效衰减、加大全部 dropout；不冻结任何塔。

与 N8 的区别：
- N8：lr×0.5、wd×2（lr×wd 不变）、首次降 lr 冻结波形塔；
- N9：lr×0.5、wd×4（AdamW 实际衰减 lr×wd 每次翻倍）、所有 nn.Dropout 的 p 每次 +0.1（上限 0.5），
  全部参数继续训练。
设计约束同 N8：文件不在 main_v2/code/ 下，不进入 N0–N7 签名范围；也不修改 n8/ 下任何文件，
N8 已完成 run 的签名不受影响；首次加大正则之前不改变任何参数、随机流或调度，前缀应与 N0 逐位一致。
"""
import csv
from pathlib import Path

import torch
import regularization_control as rc

# N9 预先写定的参数；写入签名、protocol.json 与运行记录。
SETTINGS = dict(
    plateau_patience=3,           # 验证早停分数连续 3 个整轮未超过平台最佳 + min_delta
    lr_factor=0.5,                # 每次 lr 减半
    weight_decay_factor=4.0,      # wd×4：AdamW 实际衰减 lr×wd 每次 ×2
    dropout_increment=0.1,        # 所有 p>0 的 nn.Dropout 每次 +0.1
    dropout_max=0.5,              # dropout 上限
    max_steps=3,                  # 最多加大 3 次
    freeze_any_tower=False,       # 不冻结
    plateau_signal='validation_current_epoch early-stop score (whole epochs only)',
)

# 运行时共享状态：由 run_n9.worker 设置输出目录与 min_delta。
STATE = dict(controller=None, pending_payload=None, epoch=0, min_delta=None, output_dir=None)


class N9Controller(rc.Controller):
    """在原 Controller（mode='none'）基础上追加平台触发的正则加码；不触碰 EMA/cosine 分支。"""

    def __init__(self, model, optimizer, steps_per_epoch, mode):
        if mode != 'none':
            raise ValueError(f'N9 只能叠加在 regularization_mode=none 上，收到 {mode}')
        super().__init__(model, optimizer, steps_per_epoch, mode)
        # 只调整原本启用的 dropout（p>0），不新增 dropout 位置。
        self.dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
        if not self.dropouts:
            raise RuntimeError('模型中没有可调整的 nn.Dropout')
        self.plateau_best = float('-inf')
        self.bad_epochs = 0
        self.steps_applied = 0
        self.apply_pending = False
        self.schedule_rows = []
        STATE['controller'] = self
        STATE['epoch'] = 0

    def before_step(self):
        # 平台判定在整轮验证后完成，此处在下一轮第一个优化步之前生效。
        if self.apply_pending:
            self.apply_pending = False
            for group in self.optimizer.param_groups:
                group['lr'] *= SETTINGS['lr_factor']
                group['weight_decay'] *= SETTINGS['weight_decay_factor']
            for module in self.dropouts:
                module.p = min(SETTINGS['dropout_max'], round(module.p + SETTINGS['dropout_increment'], 6))
            self.steps_applied += 1
        super().before_step()

    def observe_validation(self, score):
        """记录本轮验证分数，更新平台计数，并决定下一轮是否加大正则。"""
        STATE['epoch'] += 1
        group = self.optimizer.param_groups[0]
        improved = score > self.plateau_best + STATE['min_delta']
        if improved:
            self.plateau_best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        scheduled = False
        if self.bad_epochs >= SETTINGS['plateau_patience'] and self.steps_applied < SETTINGS['max_steps']:
            self.apply_pending = True
            self.bad_epochs = 0
            scheduled = True
        probs = sorted({round(m.p, 6) for m in self.dropouts})
        self.schedule_rows.append(dict(
            epoch=STATE['epoch'], early_stop_score=score, plateau_best=self.plateau_best,
            improved=improved, bad_epochs_after=self.bad_epochs,
            lr_used=group['lr'], weight_decay_used=group['weight_decay'],
            effective_decay_used=group['lr'] * group['weight_decay'],
            dropout_used='/'.join(f'{p:g}' for p in probs), n_dropout_modules=len(self.dropouts),
            steps_before_next_epoch=self.steps_applied + int(scheduled),
            regularization_increase_scheduled_for_next_epoch=scheduled))
        path = Path(STATE['output_dir']) / 'n9_schedule.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.schedule_rows[0]))
            writer.writeheader()
            writer.writerows(self.schedule_rows)


def install(t, output_dir, min_delta):
    """把 N9 挂到已导入的 train_sagm_net 模块 t 上；仅在 N9 worker 进程内调用。"""
    STATE.update(output_dir=str(output_dir), min_delta=float(min_delta))
    rc.Controller = N9Controller
    original_evaluate = t.evaluate_payload
    original_snapshot = t.early_stop_snapshot

    def evaluate_payload(pred, tasks, *args, **kwargs):
        payload = original_evaluate(pred, tasks, *args, **kwargs)
        source = kwargs.get('threshold_source', args[1] if len(args) > 1 else None)
        if source == 'validation_current_epoch':
            STATE['pending_payload'] = payload
        return payload

    def early_stop_snapshot(payload, *args, **kwargs):
        result = original_snapshot(payload, *args, **kwargs)
        # 只把整轮验证分数（与早停同一分数）交给平台判定；训练诊断、半轮诊断与最终评价不参与。
        if payload is STATE['pending_payload']:
            STATE['pending_payload'] = None
            STATE['controller'].observe_validation(float(result[0]))
        return result

    t.evaluate_payload = evaluate_payload
    t.early_stop_snapshot = early_stop_snapshot
