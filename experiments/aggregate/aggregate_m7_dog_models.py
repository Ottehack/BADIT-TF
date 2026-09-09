#!/usr/bin/env python3
"""Combine three small-model M7-DOG results for TPAMI Table VIII."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean,stdev
ROOT=Path(__file__).resolve().parents[2]
MODELS={"Qwen3-4B":"qwen3_4b","Llama3-3B":"llama3_3b","Gemma2-2B":"gemma2_2b"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/raw_results/m7_dog_stability_small_models_v1');a=p.parse_args();rows={};sources=[]
 for model,slug in MODELS.items():
  path=ROOT/f'experiments/raw_results/m7_dog_stability_{slug}_v1_r1/result.json';x=json.loads(path.read_text())
  if x['status']!='complete' or x['model']!=model or not all(x['assertions'].values()):raise AssertionError(model)
  rows[model]=x['metrics'];sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
 metrics={}
 for key in ('task_split_ari','seed_ari','batch_ari','cross_layer_cosine'):
  v=[rows[m][key]['mean'] for m in MODELS];metrics[key]={'mean':mean(v),'sample_std':stdev(v),'n':3,'values':v}
 out={'schema_version':1,'experiment_id':'M7-DOG-STABILITY-SMALL-MODELS-V1','status':'complete','generated_at_utc':datetime.now(timezone.utc).isoformat(),'models':rows,'macro':metrics,'sources':sources,'assertions':{'three_models':len(rows)==3,'all_finite':all(__import__('math').isfinite(z) for r in metrics.values() for z in r['values']),'official_test_not_loaded':True}}
 a.output_dir.mkdir(parents=True,exist_ok=True);path=a.output_dir/'result.json';path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'result':str(path.relative_to(ROOT)),'sha256':sha(path)},sort_keys=True))
if __name__=='__main__':main()
