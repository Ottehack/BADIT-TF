#!/usr/bin/env python3
"""Aggregate paired Table-XI timing replays without using test metrics."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SLUGS=['qwen3_8b','qwen3_4b','llama3_8b','llama3_3b','gemma2_9b','gemma2_2b']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(slug):
 idx=ROOT/f'experiments/configs/m9/table_xi_cost_v1/table_xi_cost_{slug}_v1.index.json';x=json.loads(idx.read_text());arms={};sources=[]
 for a in x['arms']:
  p=ROOT/f"experiments/raw_results/{a['run_id']}/result.json";r=json.loads(p.read_text());assert r['status']=='complete' and all(r['metrics']['assertions'].values());assert r['protocol']['official_test_loaded'] is False and r['protocol']['final_test_used_for_selection'] is False;arms[a['arm']]=float(r['timing']['training_wall_seconds_max_rank']);sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
 ratio=(float(x['calibration_wall_seconds'])+float(x['assignment_solve_seconds'])+arms['tf'])/arms['loramoe'];return {'model':x['model'],'model_slug':slug,'status':'complete','arm_order':x['arm_order'],'sft_wall_seconds':arms,'calibration_wall_seconds':x['calibration_wall_seconds'],'assignment_solve_seconds':x['assignment_solve_seconds'],'badit_tf_total_over_loramoe':ratio,'sources':sources,'assertions':{'two_complete_arms':set(arms)=={'tf','loramoe'},'test_closed':True,'positive_timings':all(v>0 for v in arms.values()),'finite_positive_ratio':ratio>0}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-slug',choices=[*SLUGS,'all'],required=True);a=p.parse_args()
 if a.model_slug!='all':r=one(a.model_slug);out=ROOT/f'experiments/raw_results/table_xi_cost_{a.model_slug}_v1'
 else:
  models={s:one(s) for s in SLUGS};vals=[v['badit_tf_total_over_loramoe'] for v in models.values()];r={'schema_version':1,'experiment_id':'TABLE-XI-TOTAL-COST-V1','status':'complete','generated_at_utc':datetime.now(timezone.utc).isoformat(),'models':models,'average':statistics.mean(vals),'assertions':{'six_models':len(models)==6,'all_assertions':all(all(v['assertions'].values()) for v in models.values())}};out=ROOT/'experiments/raw_results/table_xi_cost_all_models_v1'
 out.mkdir(parents=True,exist_ok=True);path=out/'result.json';path.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'result':str(path.relative_to(ROOT)),'sha256':sha(path)},sort_keys=True))
if __name__=='__main__':main()
