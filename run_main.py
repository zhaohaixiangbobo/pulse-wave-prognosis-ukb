"""公开版主模型入口：显式数据路径、冻结划分和N9调度；默认预览，不依赖本机历史目录。"""
from pathlib import Path
import argparse,functools,json,os,sys
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['data','p4205-cache','split-assignments','output-dir']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--evaluate-test',action='store_true')
 a=p.parse_args();args=json.loads((ROOT/'configs/n9_arguments.json').read_text())
 for flag,val in [('data',a.data),('p4205-cache',a.p4205_cache),('split-assignments',a.split_assignments),('output-dir',a.output_dir),('seed',a.seed),('train-seed',a.seed)]:args+=['--'+flag,str(val)]
 print(json.dumps({'args':args,'execute':a.execute,'evaluate_test':a.evaluate_test},indent=2),flush=True)
 if not a.execute:return
 for file in [a.data,a.p4205_cache,a.split_assignments]:
  if not file.is_file():raise FileNotFoundError(file)
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError('Use an empty output directory')
 a.output_dir.mkdir(parents=True,exist_ok=True);os.environ['REVISION_VARIANT']='main'
 sys.path.insert(0,str(ROOT/'src'));import train_sagm_net as t;import n9_control
 t.WaveformConfig=functools.partial(t.WaveformConfig,percentile_precision='numpy_2_4_1')
 original=t.parse_args
 def parse():
  parsed=original();parsed.cooperation_validation_only=not a.evaluate_test;parsed.diagnostic_half_epoch=False;parsed.regularization_mode='none'
  n9_control.install(t,parsed.output_dir,parsed.min_delta);return parsed
 t.parse_args=parse
 (a.output_dir/'public_protocol.json').write_text(json.dumps({'args':args,'n9':n9_control.SETTINGS,'test_evaluation':a.evaluate_test},indent=2),encoding='utf-8')
 sys.argv=['train_sagm_net',*args];t.main()
if __name__=='__main__':main()
