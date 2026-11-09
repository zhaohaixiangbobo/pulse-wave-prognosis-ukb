"""最终树模型公开入口：RF T1、XGBoost T1、LightGBM T2；验证集选轮，输出独立目录。"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent
CHANGES={'rf':dict(max_depth=8,min_samples_leaf=50),'xgboost':dict(max_depth=3,min_child_weight=40.,reg_lambda=10.),'lightgbm':dict(num_leaves=7,max_depth=3,min_child_samples=240,reg_lambda=10.,subsample_freq=1)}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['data','p4205-cache','split-assignments','output-dir']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--model',choices=CHANGES,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--evaluate-test',action='store_true');a=p.parse_args()
 print(json.dumps({'model':a.model,'changes':CHANGES[a.model],'execute':a.execute}),flush=True)
 if not a.execute:return
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
 a.output_dir.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT/'src'))
 import numpy as np,pandas as pd,joblib
 import train_structured_comparison_baselines as t
 from tree_diagnostics import fit_with_history
 from sklearn.metrics import average_precision_score,roc_auc_score
 f=t.load_data(a.data,10.,None);f=t.apply_cohort_filter(f,'all_p4205_valid',a.p4205_cache);idx=t.load_frozen_split_indices(f,a.split_assignments);y=f.target.to_numpy(dtype=np.int8)
 sf=t.build_structured_frame(f,feature_set='clinical_no_p4205');pre,_,_=t.build_tabular_preprocessor(sf.iloc[idx['train']]);xt=pre.fit_transform(sf.iloc[idx['train']]).astype(np.float32);xv=pre.transform(sf.iloc[idx['validation']]).astype(np.float32)
 model=t.make_model(a.model,y[idx['train']],a.seed+{'rf':1,'xgboost':2,'lightgbm':3}[a.model]*1009);model.set_params(**CHANGES[a.model]);fit_with_history(model,a.model,xt,y[idx['train']],xv,y[idx['validation']],a.output_dir)
 h=pd.read_csv(a.output_dir/'learning_history.csv');h['score']=(h.val_pr_auc+h.val_roc_auc)/2
 selected=int(h.iloc[-1].round_or_trees) if a.model=='rf' else int(h.sort_values(['score','round_or_trees'],ascending=[False,True]).iloc[0].round_or_trees)
 kw={} if a.model=='rf' else ({'iteration_range':(0,selected)} if a.model=='xgboost' else {'num_iteration':selected})
 metrics={'selected_iterations':selected,'model':a.model,'seed':a.seed}
 for split in (['validation','test'] if a.evaluate_test else ['validation']):
  ix=idx[split];x=pre.transform(sf.iloc[ix]).astype(np.float32);prob=model.predict_proba(x,**kw)[:,1]
  metrics[split]={'ap':float(average_precision_score(y[ix],prob)),'roc_auc':float(roc_auc_score(y[ix],prob))}
  pd.DataFrame({'eid':f.eid.iloc[ix].to_numpy(),'all_cause_true':y[ix],'all_cause_probability':prob}).to_csv(a.output_dir/f'{split}_predictions.csv',index=False)
 joblib.dump(pre,a.output_dir/'preprocessor.joblib');(a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
 print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
