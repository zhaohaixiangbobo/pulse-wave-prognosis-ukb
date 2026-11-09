"""从经授权的本地源表生成精确日期标签及工程特征缓存；不生成或分发参与者划分。"""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cutoff',default='2024-08-31');p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print('Preview: exact-date labels + advanced waveform feature cache. Add --execute to process local authorized data.');return
 import pandas as pd
 from exact_labels import derive_labels
 from p4205_advanced_features import build_advanced_feature_frame
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
 frame=pd.read_csv(a.input,low_memory=False)
 if frame.eid.isna().any() or frame.eid.duplicated().any():raise ValueError('Invalid participant IDs')
 labels=derive_labels(frame,a.cutoff);frame=frame.join(labels);frame=frame[frame.exclusion_reason.eq('eligible')].copy()
 a.output_dir.mkdir(parents=True,exist_ok=True);frame.to_csv(a.output_dir/'cohort.csv',index=False)
 parts=[];total=len(frame)
 for start in range(0,total,1000):
  chunk=frame.iloc[start:start+1000];feat=build_advanced_feature_frame(chunk.p4205_i0);feat.insert(0,'eid',chunk.eid.to_numpy());parts.append(feat)
  print(f'Features {min(start+1000,total)}/{total}',flush=True)
 pd.concat(parts,ignore_index=True).to_csv(a.output_dir/'p4205_advanced_features.csv',index=False)
 print('Done. Supply your authorized frozen split CSV separately. See docs/DATA_SCHEMA.md.')
if __name__=='__main__':main()
