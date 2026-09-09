#!/usr/bin/env python3
"""Freeze the 6x15 single-task controls required by mixed Forward."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2]
MODELS={
'Qwen3-8B':('qwen3_8b','experiments/configs/m0_m1_recoveries/m0_qwen3_8b_mixed_tf_seed1_r1.yaml'),
'Qwen3-4B':('qwen3_4b','experiments/configs/m0_m1_final/m0_qwen3_4b_mixed_tf_seed1.yaml'),
'Llama3-8B':('llama3_8b','experiments/configs/m0_m1_recoveries/m0_llama3_8b_mixed_tf_seed1_r1.yaml'),
'Llama3-3B':('llama3_3b','experiments/configs/m0_m1_recoveries/m0_llama3_3b_mixed_tf_seed1_r1.yaml'),
'Gemma2-9B':('gemma2_9b','experiments/configs/m0_m1_recoveries/m0_gemma2_9b_mixed_tf_seed1_r1.yaml'),
'Gemma2-2B':('gemma2_2b','experiments/configs/m0_m1_recoveries/m0_gemma2_2b_mixed_tf_seed1_r1.yaml')}
SEED=24001
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def task_slug(task):return task.split('_',1)[0]
def main():
 out=ROOT/'experiments/configs/table_i_single_task_v1';splitdir=ROOT/'experiments/configs/splits/table_i_single_task_v1';out.mkdir(parents=True,exist_ok=True);splitdir.mkdir(parents=True,exist_ok=True);records=[]
 for model,(slug,base_rel) in MODELS.items():
  base_path=ROOT/base_rel;base=yaml.safe_load(base_path.read_text());source=json.loads((ROOT/base['order_manifest']).read_text());tasks=sorted({r['task'] for r in source['train_order']});model_records=[]
  if len(tasks)!=15:raise AssertionError(model)
  for task_index,task in enumerate(tasks):
   train=[r for r in source['train_order'] if r['task']==task];test=[r for r in source['test_records'] if r['task']==task];rng=np.random.default_rng(SEED+task_index*100003);rng.shuffle(train);original=len(train);padding=(-original)%8;train=train+[train[i%original] for i in range(padding)]
   train_ids=[r['sample_id'] for r in train];unique_ids=sorted(set(train_ids));test_ids=[r['sample_id'] for r in test]
   manifest={'schema_version':1,'purpose':'Table I mixed Forward single-task control; no selection','seed':SEED,'task':task,'world_size':8,'train_order':train,'test_records':test,'original_train_count':original,'padded_train_count':len(train),'padding_repeat_count':padding,'test_count':len(test),'evaluation_role':'official_test','official_test_loaded_by_training_runner':True,'official_test_used_for_selection':False,'train_unique_ids_sha256':canonical(unique_ids),'train_order_ids_sha256':canonical(train_ids),'test_ids_sha256':canonical(test_ids),'source_mixed_manifest':base['order_manifest'],'source_mixed_manifest_sha256':sha(ROOT/base['order_manifest']),'assertions':{'one_training_task':True,'one_test_task':True,'train_test_task_match':True,'official_test_used_for_selection':False}}
   manifest['manifest_sha256']=canonical({k:v for k,v in manifest.items() if k!='manifest_sha256'});mp=splitdir/f'{slug}_{task_slug(task)}.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
   rid=f'table_i_single_task_{slug}_{task_slug(task)}_seed{SEED}_v1';c=dict(base);c.update(experiment_id='TABLE-I-MIXED-FORWARD-SINGLE-TASK-V1',todo_id='TPAMI-I',run_id=rid,seed=SEED,order_manifest=str(mp.relative_to(ROOT)),setting='mixed',assignment_method='tf',official_test_loaded=True,official_test_used_for_selection=False,table_i_single_task_control=True,table_i_task=task,table_i_control_seed=SEED,m0_m1_final=False)
   cp=out/f'{rid}.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False));row={'task':task,'task_slug':task_slug(task),'run_id':rid,'config':str(cp.relative_to(ROOT)),'config_sha256':sha(cp),'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),'train_examples':original,'test_examples':len(test)};records.append({'model':model,'model_slug':slug,**row});model_records.append(row)
  index=out/f'table_i_single_task_{slug}_seed{SEED}_v1.index.json';index.write_text(json.dumps({'schema_version':1,'model':model,'model_slug':slug,'seed':SEED,'records':model_records,'assertions':{'fifteen_tasks':len(model_records)==15,'unique_run_ids':len({r['run_id'] for r in model_records})==15}},indent=2,sort_keys=True)+'\n')
 protocol=ROOT/'experiments/materials/table_i_single_task_control_protocol.json';protocol.write_text(json.dumps({'schema_version':1,'experiment_id':'TABLE-I-MIXED-FORWARD-SINGLE-TASK-V1','definition':'per mixed seed: macro task Rouge-L minus mean of 15 fixed-seed task-specific TF controls; report five-seed mean and sample std','control_seed':SEED,'records':records,'assertions':{'six_models':len(MODELS)==6,'ninety_controls':len(records)==90,'official_test_not_selection':True,'same_tf_architecture_and_selected_model_hparams':True}},indent=2,sort_keys=True)+'\n');print(protocol,sha(protocol))
if __name__=='__main__':main()
