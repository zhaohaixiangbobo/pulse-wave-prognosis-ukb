"""公开版评价入口：验证拟合Platt校准，测试计算排序、Brier、固定阈值与DCA。"""
from pathlib import Path
import argparse,json
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import numpy as np,pandas as pd
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,balanced_accuracy_score,precision_score,recall_score,f1_score
 v=pd.read_csv(a.validation,dtype={'eid':str}).sort_values('eid');t=pd.read_csv(a.test,dtype={'eid':str}).sort_values('eid')
 for f in [v,t]:
  if f.eid.isna().any() or f.eid.duplicated().any():raise ValueError('Missing/duplicate IDs')
  if not f.all_cause_true.isin([0,1]).all():raise ValueError('Invalid labels')
  if not np.isfinite(f.all_cause_probability).all() or not f.all_cause_probability.between(0,1).all():raise ValueError('Invalid probabilities')
 if set(v.eid.astype(str)) & set(t.eid.astype(str)):raise ValueError('Validation/test overlap')
 logit=lambda x: np.log(np.clip(x,1e-6,1-1e-6)/(1-np.clip(x,1e-6,1-1e-6))).reshape(-1,1)
 cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(logit(v.all_cause_probability.to_numpy()),v.all_cause_true)
 qv=cal.predict_proba(logit(v.all_cause_probability.to_numpy()))[:,1];qt=cal.predict_proba(logit(t.all_cause_probability.to_numpy()))[:,1]
 thresholds=np.unique(np.quantile(qv,np.linspace(.05,.95,91)));best=max(thresholds,key=lambda z:balanced_accuracy_score(v.all_cause_true,qv>=z))
 y=t.all_cause_true.to_numpy();praw=t.all_cause_probability.to_numpy();k=int(np.ceil(.05*len(t)))
 order=np.argsort(t.eid.astype(str).to_numpy(),kind='stable');top=order[np.argsort(-praw[order],kind='stable')[:k]]
 result={'ap':float(average_precision_score(y,praw)),'roc_auc':float(roc_auc_score(y,praw)),'brier_score':float(brier_score_loss(y,qt)),'top5_precision':float(y[top].mean()),'threshold_validation_selected':float(best),'threshold_metrics':[]}
 for z in [best,.01,.03,.05,.10]:
  pred=qt>=z;tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum())
  result['threshold_metrics'].append({'threshold':float(z),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'net_benefit':float((tp-fp*z/(1-z))/len(y))})
 if a.output.exists():raise FileExistsError(a.output)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
