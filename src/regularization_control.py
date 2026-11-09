"""独立训练机制：按优化步调度、EMA参数与浮点缓冲平均、统一未加权损失。"""
import copy,math
import numpy as np
import torch

def lr_factor(step, steps_per_epoch):
    # 前2轮从一个步长线性升温，之后18轮cosine下降；第20轮后保持下限。
    warm=2*steps_per_epoch;end=20*steps_per_epoch;floor=1e-6/2e-4
    if step<=warm:return step/warm
    progress=min(1.,(step-warm)/(end-warm))
    return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))

class Controller:
    def __init__(self,model,optimizer,steps_per_epoch,mode):
        self.optimizer=optimizer;self.steps_per_epoch=steps_per_epoch;self.mode=mode;self.steps=0;self.active=True
        self.base_lr=optimizer.param_groups[0]['lr'];self.lrs=[]
        self.ema=copy.deepcopy(model) if mode=='ema' else None
        if self.ema is not None:
            self.ema.requires_grad_(False);self.ema.eval()

    def before_step(self):
        if self.mode=='cosine':
            for g in self.optimizer.param_groups:g['lr']=self.base_lr*lr_factor(self.steps+1,self.steps_per_epoch)
        self.lrs.append(self.optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    def after_step(self,model):
        if self.ema is not None:
            source=model.state_dict()
            for k,v in self.ema.state_dict().items():
                if self.steps==0 or not v.is_floating_point():v.copy_(source[k])
                else:v.mul_(.999).add_(source[k],alpha=.001)
        self.steps+=1

    def evaluated(self,model):return self.ema if self.active and self.ema is not None else model

def unweighted_bce(pred):
    p=np.clip(np.asarray(pred['probability'],dtype=float),1e-7,1-1e-7)
    y=np.asarray(pred['label']);m=np.asarray(pred['mask'])
    return float((-(y*np.log(p)+(1-y)*np.log1p(-p))*m).sum()/m.sum())
