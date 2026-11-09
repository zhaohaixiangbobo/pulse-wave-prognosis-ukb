"""N9 调度 CPU 单元检查（不训练真实模型、不读写 main_v2 原有目录）。

检查：
1. 首次加大正则之前，N9Controller 与原 Controller(mode='none') 的参数轨迹逐位一致；
2. 平台 3 轮后下一轮生效：lr×0.5、wd×4（lr×wd×2）、dropout +0.1（上限 0.5），最多 3 次；
3. 不冻结：加大正则后所有层（含波形塔）继续更新；
4. install() 只截获 validation_current_epoch 的早停分数。
"""
from pathlib import Path
import sys, types

sys.dont_write_bytecode = True
N9 = Path(__file__).resolve().parent
sys.path.insert(0, str(N9.parent / 'src'))
sys.path.insert(0, str(N9))
import torch
import regularization_control as rc
import n9_control

OUT = N9 / 'outputs/n9_smoke'
OUT.mkdir(parents=True, exist_ok=True)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_tower = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(.2))
        self.head = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(4, 1))

    def forward(self, x):
        return self.head(self.wave_tower(x))


def run(controller_cls, scores, steps_per_epoch=3):
    torch.manual_seed(0)
    model = Toy(); opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    n9_control.STATE.update(output_dir=str(OUT), min_delta=.0005)
    control = controller_cls(model, opt, steps_per_epoch, 'none'); model.regularization_control = control
    data = torch.randn(steps_per_epoch * 8, 4); y = torch.rand(steps_per_epoch * 8, 1)
    snapshots = []
    for score in scores:
        model.train()
        for b in range(steps_per_epoch):
            control.before_step(); opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(data[b * 8:(b + 1) * 8]), y[b * 8:(b + 1) * 8])
            loss.backward(); opt.step(); control.after_step(model)
        snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
        if hasattr(control, 'observe_validation'): control.observe_validation(score)
    return control, model, snapshots


scores = [.40, .45] + [.45] * 12
_, _, ref = run(rc.Controller, scores)
n9, model, got = run(n9_control.N9Controller, scores)
rows = n9.schedule_rows
first = next(r['epoch'] for r in rows if r['regularization_increase_scheduled_for_next_epoch'])
assert first == 5, first
for e in range(first):
    assert all(torch.equal(ref[e][k], got[e][k]) for k in ref[e]), f'epoch{e + 1} 前缀不一致'
lrs = [r['lr_used'] for r in rows]; wds = [r['weight_decay_used'] for r in rows]
decay = [r['effective_decay_used'] for r in rows]; drops = [r['dropout_used'] for r in rows]
assert abs(lrs[5] - 1e-4) < 1e-15 and abs(wds[5] - .04) < 1e-15 and abs(decay[5] - 4e-6) < 1e-15
assert drops[:5] == ['0.2'] * 5 and drops[5] == '0.3' and drops[-1] == '0.5', drops
assert n9.steps_applied == 3 and abs(decay[-1] - 1.6e-5) < 1e-15
assert all(p.requires_grad for p in model.parameters()), '不应冻结任何参数'
for prefix in ['wave_tower.0.weight', 'head.1.weight']:
    assert not torch.equal(got[first][prefix], got[-1][prefix]), f'{prefix} 加大正则后应继续更新'

# install() 截获逻辑：只把 validation_current_epoch 的分数交给控制器。
fake = types.SimpleNamespace(evaluate_payload=lambda pred, tasks, thresholds=None, threshold_source='current_split': {'src': threshold_source},
                             early_stop_snapshot=lambda payload, tasks, agg, metric: (0.5, 0.1, 0.9))
seen = []
class Probe:
    def observe_validation(self, score): seen.append(score)
original_controller = rc.Controller
n9_control.install(fake, OUT, .0005); n9_control.STATE['controller'] = Probe()
for source in ['training_diagnostic_only', 'validation_current_epoch', 'validation', 'diagnostic_only']:
    fake.early_stop_snapshot(fake.evaluate_payload(None, [], threshold_source=source), [], 'all_cause', 'mean_pr_roc')
assert seen == [0.5], seen
assert rc.Controller is n9_control.N9Controller
rc.Controller = original_controller
print(f'N9 单元检查通过：首次加大正则于epoch{first}后生效，前{first}轮与N0逐位一致；lr={lrs}；wd={wds}；dropout={drops}', flush=True)
