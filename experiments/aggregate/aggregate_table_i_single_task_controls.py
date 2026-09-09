#!/usr/bin/env python3
"""Aggregate one model's 15 single-task controls; optionally combine all models."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
MODELS={'qwen3_8b':'Qwen3-8B','qwen3_4b':'Qwen3-4B','llama3_8b':'Llama3-8B','llama3_3b':'Llama3-3B','gemma2_9b':'Gemma2-9B','gemma2_2b':'Gemma2-2B'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def model_result(slug):
 index=ROOT/f'experiments/configs/table_i_single_task_v1/table_i_single_task_{slug}_seed24001_v1.index.json';x=json.loads(index.read_text());rows=[];sources=[]
 for r in x['records']:
  path=ROOT/f"experiments/raw_results/{r['run_id']}/result.json";p=json.loads(path.read_text())
  if p['status']!='complete' or not all(p['metrics']['assertions'].values()) or p['protocol']['final_test_used_for_selection'] is not False:raise AssertionError(r['run_id'])
  per=p['metrics']['rouge']['per_task'];
  if list(per)!=[r['task']]:raise AssertionError(f"{r['run_id']} task coverage")
  rows.append({'task':r['task'],'run_id':r['run_id'],'rougeL':float(per[r['task']]['rougeL']),'n':int(per[r['task']]['n'])});sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
 baseline=statistics.mean(r['rougeL'] for r in rows);official_path=ROOT/'analysis/tables/m0_m1_official_aggregate.json';official=json.loads(official_path.read_text());cell=official['cells'][f"{MODELS[slug]}.mixed.tf"]
 mixed_rows=[{'seed':int(r['seed']),'run_id':r['terminal_registry_run_id'],'macro_rouge_l':float(r['values']['macro_rouge_l'])} for r in cell['per_seed']];values=[r['macro_rouge_l']-baseline for r in mixed_rows]
 sources.append({'path':str(official_path.relative_to(ROOT)),'sha256':sha(official_path),'bytes':official_path.stat().st_size})
 return {'model':MODELS[slug],'model_slug':slug,'status':'complete','control_seed':24001,'definition':'per mixed seed macro task Rouge-L minus fixed mean of 15 task-specific controls','single_task_baseline_mean':baseline,'single_task_rows':rows,'mixed_rows':mixed_rows,'mixed_forward':{'mean':statistics.mean(values),'sample_std':statistics.stdev(values),'n':5,'values':values},'sources':sources,'assertions':{'fifteen_controls':len(rows)==15,'five_mixed_seeds':len(values)==5,'official_test_not_selection':True}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-slug',choices=[*MODELS,'all'],required=True);a=p.parse_args()
 if a.model_slug!='all':
  result=model_result(a.model_slug);out=ROOT/f"experiments/raw_results/table_i_single_task_{a.model_slug}_seed24001_v1";out.mkdir(parents=True,exist_ok=True);path=out/'result.json'
 else:
  models={slug:model_result(slug) for slug in MODELS};result={'schema_version':1,'experiment_id':'TABLE-I-MIXED-FORWARD-SINGLE-TASK-V1','status':'complete','generated_at_utc':datetime.now(timezone.utc).isoformat(),'models':models,'assertions':{'six_models':len(models)==6,'all_model_assertions':all(all(x['assertions'].values()) for x in models.values())}};out=ROOT/'experiments/raw_results/table_i_single_task_all_models_v1';out.mkdir(parents=True,exist_ok=True);path=out/'result.json'
 path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'result':str(path.relative_to(ROOT)),'sha256':sha(path)},sort_keys=True))
if __name__=='__main__':main()
