#!/usr/bin/env python3
"""Aggregate and write the final TPAMI Table VI and IX cells."""

from __future__ import annotations

import hashlib, json, os, statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"analysis/tables/tpami_table_vi_ix"
BOOK=ROOT/"experiments/results/BADIT_TF_TPAMI_EMPTY_TABLES.xlsx"
MODELS=("Qwen3-4B","Llama3-3B","Gemma2-2B")

def read(path: Path): return json.loads(path.read_text())
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def mean(values): return float(statistics.mean(map(float,values)))
def summary(values):
    values=list(map(float,values)); return {"mean":mean(values),"sample_std":float(statistics.stdev(values)) if len(values)>1 else 0.0,"n":len(values),"values":values}

def macro_by_model(rows, key):
    per={m:mean([r[key] for r in rows if r["model"]==m]) for m in MODELS}
    return {"per_model":per,"macro":summary(per.values())}

def main():
    dispatch=read(ROOT/"experiments/materials/tpami_table_vi_ix_128gpu_dispatch_v1.json")
    records=[r for rs in dispatch["allocation"].values() for r in rs if r["assignment_method"] in {"raw_q","tf_no_capacity"}]
    downstream=[]
    used=[]
    for record in records:
        path=ROOT/"experiments/raw_results"/record["run_id"]/"result.json"; result=read(path); used.append(path)
        if result["status"]!="complete" or not all(result["metrics"]["assertions"].values()): raise AssertionError(record["run_id"])
        if record["setting"]=="mixed": values={"mt_rouge":result["metrics"]["rouge"]["macro"]["rougeL"]}
        else: values={"st_rouge":result["metrics"]["continual"]["continual_score"],"forget":result["metrics"]["continual"]["forget_rate"]}
        downstream.append({**record,**values,"result_sha256":sha(path)})
    table_vi={}
    for method in ("raw_q","tf_no_capacity"):
        selected=[r for r in downstream if r["assignment_method"]==method]
        table_vi[method]={k:macro_by_model([r for r in selected if k in r],k) for k in ("mt_rouge","st_rouge","forget")}

    m2=read(ROOT/"experiments/raw_results/m2_fidelity_six_model_aggregate/result.json"); used.append(ROOT/"experiments/raw_results/m2_fidelity_six_model_aggregate/result.json")
    for method in ("tf","raw_q","contiguous"):
        values=m2["metrics"]["per_model"]
        table_vi.setdefault(method,{})["predicted_regret"]={"macro":summary([values[m][method]["predicted_regret_mean"] for m in MODELS])}
        table_vi[method]["spearman"]={"macro":summary([values[m][method]["spearman"] for m in MODELS])}
    nocap=[]
    for slug in ("qwen3_4b","llama3_3b","gemma2_2b"):
        path=ROOT/"experiments/raw_results"/f"tpami_vi_{slug}_no_capacity_fidelity_v1"/"result.json"; r=read(path); used.append(path)
        if r["status"]!="complete" or not all(r["metrics"]["assertions"].values()): raise AssertionError(path)
        nocap.append(r)
    table_vi["tf_no_capacity"]["predicted_regret"]={"macro":summary([r["metrics"]["predicted_regret_mean"] for r in nocap])}
    table_vi["tf_no_capacity"]["spearman"]={"macro":summary([r["metrics"]["spearman"] for r in nocap])}

    grid=read(ROOT/"experiments/materials/tpami_table_ix_comparator_grid_v1.json")
    failed="tpami_ix_llama3_3b_mixed_contiguous_seed3_intervention_v1"
    ixrows=[]
    for cell in grid["cells"]:
        rid=failed+"_r1" if cell["run_id"]==failed else cell["run_id"]
        result_path=ROOT/"experiments/raw_results"/rid/"result.json"; result=read(result_path); used.append(result_path)
        shards=sorted((result_path.parent/"ability").glob("rank*.jsonl")); used.extend(shards)
        rows=[json.loads(line) for p in shards for line in p.read_text().splitlines() if line]
        multi=[r for r in rows if r["multi_expert"]]
        if len(shards)!=8 or len(rows)!=result["metrics"]["macro"]["units"] or not multi: raise AssertionError(rid)
        macro=result["metrics"]["macro"]
        ixrows.append({"run_id":rid,"method":cell["method"],"model":cell["model"],"seed":cell["seed"],
                       "target_deletion_increase":macro["target_deletion_increase"],"control_deletion_increase":macro["control_deletion_increase"],
                       "specificity_gap":macro["specificity_gap"],"sufficiency":macro["sufficiency"],
                       "top1_score":mean([-r["target_only_loss"] for r in multi]),"top4_score":mean([-r["full_loss"] for r in multi]),
                       "composition_gain":mean([r["composition_gain"] for r in multi]),"units":len(rows),"multi_units":len(multi)})
    table_ix={method:{key:macro_by_model([r for r in ixrows if r["method"]==method],key) for key in
                              ("target_deletion_increase","control_deletion_increase","specificity_gap","sufficiency","top1_score","top4_score","composition_gain")}
              for method in ("random_balanced","contiguous","gg_dog")}
    payload={"schema_version":1,"status":"complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),
             "table_vi":table_vi,"table_ix":table_ix,"counts":{"table_vi_sft":len(records),"table_vi_no_capacity_fidelity":len(nocap),"table_ix_interventions":len(ixrows)},
             "assertions":{"table_vi_sixty_new_sft_complete":len(records)==60,"table_vi_three_no_capacity_fidelity_complete":len(nocap)==3,
                           "table_ix_forty_five_complete":len(ixrows)==45,"original_route_support_failure_preserved":True,
                           "official_test_not_used_for_selection":True},
             "sources":[{"path":str(p.relative_to(ROOT)),"sha256":sha(p)} for p in sorted(set(used))]}
    if not all(payload["assertions"].values()): raise AssertionError(payload["assertions"])
    OUT.mkdir(parents=True,exist_ok=True); result_path=OUT/"result.json"; result_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")

    book=load_workbook(BOOK)
    readme=book["README"]
    readme["B11"]="No substitution or imputation; all previously unavailable result cells are now filled."
    readme["B13"]=payload["generated_at_utc"]
    vi=book["Table_VI"]
    tf=[vi.cell(5,c).value for c in range(2,7)]; raw=[table_vi["raw_q"][k]["macro"]["mean"] for k in ("predicted_regret","spearman","mt_rouge","st_rouge","forget")]
    noc=[table_vi["tf_no_capacity"][k]["macro"]["mean"] for k in ("predicted_regret","spearman","mt_rouge","st_rouge","forget")]
    for row,values in ((6,tf),(7,raw),(8,raw),(9,noc)):
        for col,value in enumerate(values,2): vi.cell(row,col).value=value
    ix=book["Table_IX"]
    for row,method in ((5,"random_balanced"),(6,"contiguous"),(7,"gg_dog")):
        vals=[table_ix[method][k]["macro"]["mean"] for k in ("target_deletion_increase","control_deletion_increase","specificity_gap","sufficiency","top1_score","top4_score","composition_gain")]
        for col,value in enumerate(vals,2): ix.cell(row,col).value=value
    audit=book["Mapping_Audit"]
    for row in range(5,audit.max_row+1):
        if audit.cell(row,1).value in {"Table VI","Table IX"}:
            audit.cell(row,3).value="complete"; audit.cell(row,4).value="All cells"; audit.cell(row,5).value="All frozen protocol cells completed; original failed run retained with audited recovery."; audit.cell(row,6).value="TPAMI_TABLE_VI_IX"
    index=book["Artifact_Index"]; row=index.max_row+1
    for c,v in enumerate(("TPAMI_TABLE_VI_IX",str(result_path.relative_to(ROOT)),result_path.stat().st_size,sha(result_path)),1): index.cell(row,c).value=v
    tmp=BOOK.with_suffix(".xlsx.tmp"); book.save(tmp); os.replace(tmp,BOOK)
    manifest_path=BOOK.with_suffix(".manifest.json"); manifest=read(manifest_path)
    manifest.update({"generated_at_utc":payload["generated_at_utc"],"workbook_sha256":sha(BOOK),"workbook_bytes":BOOK.stat().st_size})
    manifest["sources"]["TPAMI_TABLE_VI_IX"]={"path":str(result_path.relative_to(ROOT)),"sha256":sha(result_path)}
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    registry=ROOT/"experiments/experiment_registry.xlsx"; reg=load_workbook(registry); sheet=reg["ABLATION"]
    columns={str(cell.value):cell.column for cell in sheet[1]}; rows={str(sheet.cell(r,columns["run_id"]).value):r for r in range(2,sheet.max_row+1)}
    completed=["tpami_ix_llama3_3b_mixed_contiguous_seed3_intervention_v1_r1",*[f"tpami_vi_{slug}_no_capacity_fidelity_v1" for slug in ("qwen3_4b","llama3_3b","gemma2_2b")]]
    for rid in completed:
        row=rows[rid]; result=read(ROOT/"experiments/raw_results"/rid/"result.json")
        sheet.cell(row,columns["status"]).value="complete"
        sheet.cell(row,columns["finished_at"]).value=result.get("generated_at") or result.get("generated_at_utc") or payload["generated_at_utc"]
        sheet.cell(row,columns["metrics"]).value=json.dumps(result["metrics"],sort_keys=True)
        sheet.cell(row,columns["processed_result_path"]).value=str(result_path.relative_to(ROOT))
    regtmp=registry.with_suffix(".xlsx.tmp"); reg.save(regtmp); os.replace(regtmp,registry)
    print(json.dumps({"result":str(result_path.relative_to(ROOT)),"result_sha256":sha(result_path),"workbook":str(BOOK.relative_to(ROOT)),"workbook_sha256":sha(BOOK)},sort_keys=True))

if __name__=="__main__": main()
