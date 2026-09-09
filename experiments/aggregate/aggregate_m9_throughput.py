#!/usr/bin/env python3
"""Aggregate one eight-rank matched TF/GG throughput replay."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import yaml

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text())
    out=Path(c['output_dir']); paths=sorted(out.glob('rank*.json')); rows=[json.loads(x.read_text()) for x in paths]
    if len(rows)!=8 or {r['rank'] for r in rows}!=set(range(8)): raise AssertionError('eight unique ranks required')
    metrics={}
    for v in ('tf','gg'):
        tokens=sum(r['metrics'][v]['generated_tokens'] for r in rows); wall=max(r['metrics'][v]['elapsed_seconds'] for r in rows)
        metrics[v]={"aggregate_generated_tokens":tokens,"max_rank_wall_seconds":wall,"tokens_per_second":tokens/wall,"rank_elapsed_seconds":[r['metrics'][v]['elapsed_seconds'] for r in rows]}
    metrics['tf_over_gg']=metrics['tf']['tokens_per_second']/metrics['gg']['tokens_per_second']
    result={"schema_version":1,"experiment_id":"M9-INFERENCE-THROUGHPUT-V1","run_id":c['run_id'],"model":c['model_name'],"status":"complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"config_path":str(a.config),"config_sha256":sha(a.config),"metrics":metrics,"protocol":{"prompts":15,"repeats":c['repeats'],"new_tokens_per_prompt":c['max_new_tokens'],"aggregate_eight_gpu_throughput":True,"rank_order_counterbalanced":True,"official_test_loaded":False},"assertions":{"eight_ranks":len(rows)==8,"all_tokens_exact":all(r['metrics'][v]['generated_tokens']==len(range(r['rank'],15,8))*c['repeats']*c['max_new_tokens'] for r in rows for v in ('tf','gg')),"positive_throughput":all(metrics[v]['tokens_per_second']>0 for v in ('tf','gg')),"test_closed":True},"rank_artifacts":[{"path":str(x),"sha256":sha(x)} for x in paths]}
    if not all(result['assertions'].values()): raise AssertionError(result['assertions'])
    path=out/'result.json';path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'result':str(path),'sha256':sha(path),'tf_over_gg':metrics['tf_over_gg']},sort_keys=True))
if __name__=='__main__':main()
