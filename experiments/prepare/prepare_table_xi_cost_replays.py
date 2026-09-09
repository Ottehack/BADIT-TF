#!/usr/bin/env python3
"""Freeze paired one-epoch BADIT-TF/LoRAMoE SFT timing replays."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
MODELS={
'Qwen3-8B':('qwen3_8b','experiments/configs/m0_m1_recoveries/m0_qwen3_8b_mixed_tf_seed1_r1.yaml'),
'Qwen3-4B':('qwen3_4b','experiments/configs/m0_m1_final/m0_qwen3_4b_mixed_tf_seed1.yaml'),
'Llama3-8B':('llama3_8b','experiments/configs/m0_m1_recoveries/m0_llama3_8b_mixed_tf_seed1_r1.yaml'),
'Llama3-3B':('llama3_3b','experiments/configs/m0_m1_recoveries/m0_llama3_3b_mixed_tf_seed1_r1.yaml'),
'Gemma2-9B':('gemma2_9b','experiments/configs/m0_m1_recoveries/m0_gemma2_9b_mixed_tf_seed1_r1.yaml'),
'Gemma2-2B':('gemma2_2b','experiments/configs/m0_m1_recoveries/m0_gemma2_2b_mixed_tf_seed1_r1.yaml')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 post=ROOT/'analysis/tables/tpami_posthoc/result.json';cost=json.loads(post.read_text());out=ROOT/'experiments/configs/m9/table_xi_cost_v1';out.mkdir(parents=True,exist_ok=True);records=[]
 for i,(model,(slug,base_rel)) in enumerate(MODELS.items()):
  base=yaml.safe_load((ROOT/base_rel).read_text());arms=[]
  for arm in ('tf','loramoe'):
   c=dict(base);rid=f'table_xi_cost_{slug}_{arm}_seed1_v1';c.update(experiment_id='TABLE-XI-TOTAL-COST-V1',todo_id='M9',run_id=rid,output_root='experiments/raw_results',seed=1,official_test_loaded=False,official_test_used_for_selection=False,table_xi_cost_replay=True,table_xi_arm=arm)
   if arm=='loramoe':
    c.pop('initial_bank_checkpoint',None);c.pop('initial_bank_checkpoint_sha256',None);c.update(assignment_method='contiguous',initialization_method='kaiming_zero',router_mode='upstream_softmax',router_bias=False,apply_lora_dropout=False,dense_steps=0,learning_rate=0.0002,router_lr_multiplier=1.0,warmup_ratio=0.03,lora_dropout=0.05,num_experts=8,rank=4,lora_alpha=32,top_k=4)
   cp=out/f'{rid}.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False));arms.append({'arm':arm,'run_id':rid,'config':str(cp.relative_to(ROOT)),'config_sha256':sha(cp)})
  order=['tf','loramoe'] if i%2==0 else ['loramoe','tf'];idx=out/f'table_xi_cost_{slug}_v1.index.json';idx.write_text(json.dumps({'schema_version':1,'model':model,'model_slug':slug,'arm_order':order,'arms':arms,'calibration_wall_seconds':cost['table_xii_calibration_timing'][model]['calibration_wall_seconds'],'assignment_solve_seconds':cost['table_xii_assignment_timing'][model]['assignment_solve_seconds'],'cost_source':str(post.relative_to(ROOT)),'cost_source_sha256':sha(post),'assertions':{'two_arms':len(arms)==2,'same_order_manifest':len({yaml.safe_load((ROOT/a['config']).read_text())['order_manifest'] for a in arms})==1,'test_closed':True,'counterbalanced_order':True}},indent=2,sort_keys=True)+'\n');records.append({'model':model,'model_slug':slug,'index':str(idx.relative_to(ROOT)),'index_sha256':sha(idx),'arm_order':order})
 protocol=ROOT/'experiments/materials/table_xi_total_cost_protocol.json';protocol.write_text(json.dumps({'schema_version':1,'experiment_id':'TABLE-XI-TOTAL-COST-V1','definition':'(frozen Gate/Fisher calibration wall seconds + exact assignment CPU seconds + paired BADIT-TF one-epoch SFT wall seconds) / matched LoRAMoE one-epoch SFT wall seconds','timing_scope':'SFT loop only; identical frozen mixed seed-1 order, token budget, 8-GPU node; model/adapter/DeepSpeed load, eval, checkpoint and upload excluded','loramoe_contract':{'source':'uploaded LoRAMoE loramoe_k8.config and audited effective forward','num_experts':8,'rank':4,'alpha':32,'top_k':4,'router':'bias-free softmax top-k renormalized','initialization':'Kaiming-A/zero-B','effective_dropout':False},'records':records,'assertions':{'six_models':len(records)==6,'paired_arms':True,'rank_order_counterbalanced':True,'official_test_closed':True}},indent=2,sort_keys=True)+'\n');print(protocol,sha(protocol))
if __name__=='__main__':main()
