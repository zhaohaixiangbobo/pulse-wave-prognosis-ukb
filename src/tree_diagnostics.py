"""树模型固定配置训练与分阶段训练/验证诊断；不通过测试集选轮数。"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score


def fit_with_history(model, name, x_train, y_train, x_val, y_val, output):
    """每25棵/轮记录精确AP与ROC；RF为累积树预测诊断，不称为boosting。"""
    model.fit(x_train, y_train)
    total = int(model.get_params().get('n_estimators', 1))
    rows = []
    if name == 'rf':
        sums = [np.zeros(len(x_train)), np.zeros(len(x_val))]
        for i, tree in enumerate(model.estimators_, 1):
            for j, x in enumerate([x_train,x_val]):
                sums[j] += tree.predict_proba(x)[:,1]
            if i % 25 == 0 or i == total:
                rows.append(make_row(i,y_train,sums[0]/i,y_val,sums[1]/i))
                print(f'[RF诊断] {i}/{total}棵',flush=True)
    elif name in ['xgboost','lightgbm']:
        for i in sorted(set(list(range(25,total+1,25))+[total])):
            kw = {'iteration_range':(0,i)} if name == 'xgboost' else {'num_iteration':i}
            rows.append(make_row(i,y_train,model.predict_proba(x_train,**kw)[:,1],y_val,model.predict_proba(x_val,**kw)[:,1]))
            print(f'[树诊断] {name} {i}/{total}轮',flush=True)
    pd.DataFrame(rows).to_csv(output/'learning_history.csv',index=False,encoding='utf-8')
    (output/'model_parameters.json').write_text(json.dumps(model.get_params(),ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    joblib.dump(model,output/'model.joblib')
    return model


def make_row(i, yt, pt, yv, pv):
    """所有模型采用相同的排序指标实现。"""
    return {'round_or_trees':i,'train_roc_auc':roc_auc_score(yt,pt),'train_pr_auc':average_precision_score(yt,pt),
            'val_roc_auc':roc_auc_score(yv,pv),'val_pr_auc':average_precision_score(yv,pv)}
