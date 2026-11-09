"""记录裁剪前梯度和逐时点验证预测；可选半轮评价隔离随机状态。"""
from contextlib import contextmanager
import random
import numpy as np
import torch

class GradientTracker:
    def __init__(self):
        self.values=[]
        self.clipped=[]

    def observe(self, parameters, limit):
        norms=[torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters if p.grad is not None]
        norm=float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.
        if not np.isfinite(norm):raise FloatingPointError('检测到非有限梯度，停止并保留日志')
        self.values.append(norm);self.clipped.append(limit>0 and norm>limit)

    def summary(self):
        return dict(gradient_norm_preclip_mean=float(np.mean(self.values)),gradient_norm_preclip_max=float(max(self.values)),gradient_clip_fraction=float(np.mean(self.clipped)))

@contextmanager
def isolated_evaluation(model):
    """半轮诊断不改变后续dropout及数据加载所使用的随机流。"""
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    np_state=np.random.get_state();py_state=random.getstate();modes=[(m,m.training) for m in model.modules()]
    try:yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(np_state);random.setstate(py_state)
        for m,mode in modes:m.training=mode

def record_prediction(directory, epoch, steps, prediction, criterion):
    """按固定验证集顺序保存，每轮仅约数百KB；bootstrap独立后处理。"""
    directory.mkdir(parents=True,exist_ok=True)
    y=np.asarray(prediction['label']);mask=np.asarray(prediction['mask'])
    assert y.shape[1]==1 and np.all(mask==1),'本轮限定无mask的全因主任务'
    np.savez_compressed(directory/f'step_{steps:07d}.npz',epoch=float(epoch),steps=int(steps),
        probability=np.asarray(prediction['probability'])[:,0],label=y[:,0],
        weighted_loss=float(prediction['loss']),positive_weight=float(criterion.pos_weight[0].item()))
