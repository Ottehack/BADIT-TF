#!/usr/bin/env python3
"""Fill only the em-dash result targets that actually appear in TPAMI.pdf.

The PDF table schemas are treated as authoritative.  Existing paper baseline
numbers are not recopied: this workbook focuses on the rows/cells that were
blank in the supplied manuscript and maps current auditable artifacts into
those exact columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment

from export_current_paper_tables_xlsx import NEEDED, add_table, git_commit, new_sheet, read_json, sha256


MODEL_ORDER = ["Qwen3-8B", "Qwen3-4B", "Llama3-8B", "Llama3-3B", "Gemma2-9B", "Gemma2-2B"]
SMALL_MODELS = ["Qwen3-4B", "Llama3-3B", "Gemma2-2B"]
MODEL_NAMES = {
    "qwen3_8b": "Qwen3-8B", "qwen3_4b": "Qwen3-4B",
    "llama3_8b": "Llama3-8B", "llama3_3b": "Llama3-3B",
    "gemma2_9b": "Gemma2-9B", "gemma2_2b": "Gemma2-2B",
}


def pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def f(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def m6_mixed_folds(root: Path, manifest: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    grouped: dict[str, list[dict[str, Any]]] = {"four_category": [], "random12": []}
    used: list[Path] = []
    for record in manifest["records"]:
        if record["setting"] != "mixed":
            continue
        path = root / "experiments/raw_results" / record["run_id"] / "result.json"
        if not path.is_file():
            continue
        result = read_json(path)
        per_task = result["metrics"]["rouge"]["per_task"]
        grouped[record["scope"]].append(
            {
                "fold": record["fold"],
                "seen": mean(per_task[t]["rougeL"] for t in record["discovery_tasks"]),
                "unseen": mean(per_task[t]["rougeL"] for t in record["heldout_tasks"]),
                "overall": result["metrics"]["rouge"]["macro"]["rougeL"],
                "run_id": record["run_id"],
            }
        )
        used.append(path)
    return grouped, used


def stat_pm(items: list[dict[str, Any]], key: str) -> str:
    values = [x[key] for x in items]
    return pm(statistics.mean(values), statistics.stdev(values))


def weighted_group_mean(groups: list[dict[str, Any]]) -> tuple[Any, int]:
    valid = [x for x in groups if x["n"] > 0 and x["mean"] is not None]
    if not valid:
        return NEEDED, 0
    n = sum(x["n"] for x in valid)
    return sum(x["mean"] * x["n"] for x in valid) / n, n


def make_workbook(root: Path, pdf: Path, output: Path) -> dict[str, Any]:
    paths = {
        "M0_M1": root / "analysis/tables/m0_m1_official_aggregate.json",
        "M2": root / "experiments/raw_results/m2_fidelity_six_model_aggregate/result.json",
        "M3": root / "analysis/tables/m3_deployment_scale/result.json",
        "M4": root / "analysis/tables/m4_decomposition_fidelity/result.json",
        "M5": root / "analysis/tables/m5_tf_ablation_contiguous/result.json",
        "M6": root / "experiments/materials/m6_discovery_transfer_release_manifest.json",
        "M7": root / "experiments/raw_results/m7_partition_stability_small_models_formal_r1/result.json",
        "M7_DOG": root / "experiments/raw_results/m7_dog_stability_small_models_v1/result.json",
        "M8": root / "analysis/tables/m8_ability_evidence/result.json",
        "M9": root / "analysis/tables/m9_efficiency_partial.json",
        "A0": root / "analysis/tables/a0_calibration_config/result.json",
        "A1": root / "analysis/tables/a1_solver/result.json",
        "TPAMI_POSTHOC": root / "analysis/tables/tpami_posthoc/result.json",
        "TABLE_I": root / "experiments/raw_results/table_i_single_task_all_models_v1_r1/result.json",
        "TABLE_XI": root / "experiments/raw_results/table_xi_cost_all_models_v1_r1/result.json",
    }
    missing = [str(path) for path in [pdf, *paths.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    d = {name: read_json(path) for name, path in paths.items()}
    m01, m2, m3, m4, m5, m7, m7_dog, m8, m9 = (d[x] for x in ("M0_M1", "M2", "M3", "M4", "M5", "M7", "M7_DOG", "M8", "M9"))
    table_i, table_xi = d["TABLE_I"], d["TABLE_XI"]
    posthoc = d["TPAMI_POSTHOC"]
    m6_groups, m6_results = m6_mixed_folds(root, d["M6"])

    wb = Workbook()
    wb.remove(wb.active)
    audits: list[list[Any]] = []

    ws = new_sheet(wb, "README", "以用户提供的 TPAMI.pdf 为唯一表格结构依据；只整理 PDF 中 em dash 标记的待填结果。")
    add_table(ws, ["Field", "Value"], [
        ["Supplied PDF", str(pdf)],
        ["PDF SHA256", sha256(pdf)],
        ["PDF pages", 30],
        ["Blank paper tables", "Table I-IX, XI-XII, XIX-XXI (14 tables)"],
        ["Excluded", "Table X and XIII-XVIII already contain paper numbers and are not new-result blanks."],
        ["Mapping policy", "Exact PDF row/column schema; only locally auditable new experiments are inserted."],
        ["Missing policy", f"No substitution or imputation: unavailable cells remain {NEEDED}."],
        ["Important correction", "Table IX Top-1/Top-4 are performance metrics; top1_mass is not substituted."],
        ["Snapshot UTC", datetime.now(timezone.utc).isoformat()],
        ["Git commit", git_commit(root)],
    ])

    # Table I: only the six blank BADIT-TF rows from the paper.
    rows = []
    for model in MODEL_ORDER:
        mt = m01["cells"][f"{model}.mixed.tf"]["aggregate"]["macro_rouge_l"]
        st = m01["cells"][f"{model}.sequential.tf"]["aggregate"]
        rows.append([
            model, "BADIT-TF (ours)", pm(mt["mean"], mt["std"]),
            pm(table_i["models"][next(k for k, v in MODEL_NAMES.items() if v == model)]["mixed_forward"]["mean"], table_i["models"][next(k for k, v in MODEL_NAMES.items() if v == model)]["mixed_forward"]["sample_std"]),
            pm(st["continual_score"]["mean"], st["continual_score"]["std"]),
            pm(st["forget_rate"]["mean"], st["forget_rate"]["std"]),
            pm(st["forward"]["mean"], st["forward"]["std"]),
            pm(st["backward"]["mean"], st["backward"]["std"]),
        ])
    ws = new_sheet(wb, "Table_I", "PDF p.9: Mixed and sequential multi-task instruction tuning; only blank BADIT-TF rows shown.")
    add_table(ws, ["Model", "Method", "Mixed ROUGE↑", "Mixed Forward↑", "Sequential ROUGE↑", "Forget Rate↓", "Sequential Forward↑", "Backward↑"], rows)
    audits.append(["Table I", 9, "complete", "36/36 BADIT-TF metric cells", "Mixed Forward uses fixed-seed task-specific controls recovered from 90 immutable child artifacts.", "M0_M1+TABLE_I"])

    # Table II.
    rows = []
    mt_means: list[float] = []
    st_means: list[float] = []
    forget_means: list[float] = []
    mt_wins: list[int] = []
    st_wins: list[int] = []
    for model in MODEL_ORDER:
        mt = m01["paired"][f"{model}.mixed"]["tf_minus_gg"]["macro_rouge_l"]
        seq = m01["paired"][f"{model}.sequential"]["tf_minus_gg"]
        mw = sum(x > 0 for x in mt["values"])
        sw = sum(x > 0 for x in seq["continual_score"]["values"])
        ci = posthoc["table_ii"][model]["hierarchical_bootstrap_95ci"]
        rows.append([model, f(mt["mean"], 4), f"[{ci[0]:.4f}, {ci[1]:.4f}]", mw, f(seq["continual_score"]["mean"], 4), f(seq["forget_rate"]["mean"], 4), sw])
        mt_means.append(mt["mean"]); st_means.append(seq["continual_score"]["mean"]); forget_means.append(seq["forget_rate"]["mean"]); mt_wins.append(mw); st_wins.append(sw)
    ci = posthoc["table_ii"]["Macro average"]["hierarchical_bootstrap_95ci"]
    rows.append(["Macro average", f(mean(mt_means), 4), f"[{ci[0]:.4f}, {ci[1]:.4f}]", f(mean(mt_wins), 3), f(mean(st_means), 4), f(mean(forget_means), 4), f(mean(st_wins), 3)])
    ws = new_sheet(wb, "Table_II", "PDF p.10: Paired TF-GG rerun. Differences are TF minus GG; negative Delta Forget favors TF.")
    add_table(ws, ["Model", "MT ΔROUGE", "95% CI", "MT wins", "ST ΔROUGE", "ΔForget", "ST wins"], rows)
    audits.append(["Table II", 10, "complete", "All 42 cells", "10,000 hierarchical bootstrap replicates; tasks then paired seeds.", "M0_M1+TPAMI_POSTHOC"])

    # Table III.
    method_map = [("Contiguous SVD", "contiguous"), ("Random balanced", "random_balanced"), ("BADIT-GG (DOG)", "gg_dog"), ("Raw-q grouping", "raw_q"), ("BADIT-TF", "tf")]
    rows = []
    for label, key in method_map:
        x = m2["metrics"]["macro"][key]
        rows.append([label, f(x["predicted_regret_mean"]), f(x["observed_gap_mean"]), f(x["spearman"]), f(x["rank_accuracy"]), f(x["top_bottom_observed_gap_separation"])])
    ws = new_sheet(wb, "Table_III", "PDF p.11: Assignment-objective fidelity on held-out probes; locked eta=0.0001.")
    add_table(ws, ["Assignment", "Pred. regret↓", "Obs. gap↓", "Spearman↑", "Rank acc.↑", "Top-bottom gap↑"], rows)
    audits.append(["Table III", 11, "complete", "25/25", "Local eta is protocol-amended to 0.0001.", "M2"])

    # Table IV.
    rows = []
    for eta in m3["eta_grid"]:
        x = m3["macro_by_eta"][str(eta)]
        rows.append([eta, f(x["predicted_delta_loss_mean"]["mean"]), f(x["observed_delta_loss_mean"]["mean"]), f(x["spearman"]["mean"]), f(x["relative_error_median"]["mean"])])
    ws = new_sheet(wb, "Table_IV", "PDF p.11: Deployment-path fidelity; Rel. err. is the macro mean of per-model median relative error.")
    add_table(ws, ["η", "Pred. ΔL", "Obs. ΔL", "Spearman↑", "Rel. err.↓"], rows)
    audits.append(["Table IV", 11, "complete", "16/16", "Six-model macro.", "M3"])

    # Table V.
    rows = []
    for label, key in [("Contiguous SVD", "contiguous"), ("BADIT-GG (DOG)", "gg_dog"), ("BADIT-TF", "tf")]:
        x = m4["macro"][key]
        rows.append([label, f(x["grouping_mean"]["mean"]), f(x["routing_mean"]["mean"]), f(x["total_pred_mean"]["mean"]), f(x["observed_gap_mean"]["mean"])])
    ws = new_sheet(wb, "Table_V", "PDF p.11: Empirical grouping-routing decomposition audit; eta=0.0001.")
    add_table(ws, ["Assignment", "Grouping↓", "Routing↓", "Total pred.↓", "Obs. gap↓"], rows)
    audits.append(["Table V", 11, "complete", "12/12", "30 formal cells per assignment.", "M4"])

    # Table VI. Held-out objective cells are a three-small-model reduction of M2.
    m2_small: dict[str, tuple[float, float]] = {}
    for key in ("tf", "contiguous", "raw_q"):
        m2_small[key] = (
            mean(m2["metrics"]["per_model"][model][key]["predicted_regret_mean"] for model in SMALL_MODELS),
            mean(m2["metrics"]["per_model"][model][key]["spearman"] for model in SMALL_MODELS),
        )
    tf_mt = m5["macro"]["mixed"]["tf"]["macro_rouge_l"]["mean"]
    tf_st = m5["macro"]["sequential"]["tf"]
    co_mt = m5["macro"]["mixed"]["contiguous"]["macro_rouge_l"]["mean"]
    co_st = m5["macro"]["sequential"]["contiguous"]
    rows = [
        ["BADIT-TF", f(m2_small["tf"][0]), f(m2_small["tf"][1]), f(tf_mt), f(tf_st["continual_score"]["mean"]), f(tf_st["forget_rate"]["mean"])],
        ["w/o task balancing (wi=1/I)", NEEDED, NEEDED, NEEDED, NEEDED, NEEDED],
        ["w/o curvature (F_l=I)", NEEDED, NEEDED, NEEDED, NEEDED, NEEDED],
        ["Raw-q grouping", f(m2_small["raw_q"][0]), f(m2_small["raw_q"][1]), NEEDED, NEEDED, NEEDED],
        ["w/o equal capacity", NEEDED, NEEDED, NEEDED, NEEDED, NEEDED],
        ["Contiguous SVD assignment", f(m2_small["contiguous"][0]), f(m2_small["contiguous"][1]), f(co_mt), f(co_st["continual_score"]["mean"]), f(co_st["forget_rate"]["mean"])],
    ]
    ws = new_sheet(wb, "Table_VI", "PDF p.12: TF component ablations averaged over Qwen3-4B, Llama3-3B, Gemma2-2B.")
    add_table(ws, ["Variant", "Held-out regret↓", "Fidelity rho_S↑", "MT ROUGE↑", "ST ROUGE↑", "Forget↓"], rows)
    audits.append(["Table VI", 12, "partial", "TF/contiguous plus raw-q objective", "Four downstream ablations are not complete.", "M2+M5"])

    # Table VII.
    def posthoc_cell(row: str, metric: str) -> Any:
        value = posthoc["table_vii"][row].get(metric)
        return pm(value["mean"], value["sample_std"]) if isinstance(value, dict) and value.get("n") else NEEDED

    rows = [
        ["All 15 tasks", 15, posthoc_cell("all15", "seen_mt"), posthoc_cell("all15", "unseen_mt"), posthoc_cell("all15", "unseen_st"), posthoc_cell("all15", "unseen_forget"), posthoc_cell("all15", "overall_mt")],
        ["Four-category discovery (5 folds)", 12, posthoc_cell("four_category", "seen_mt"), posthoc_cell("four_category", "unseen_mt"), posthoc_cell("four_category", "unseen_st"), posthoc_cell("four_category", "unseen_forget"), posthoc_cell("four_category", "overall_mt")],
        ["Random 12-task discovery (5 splits)", 12, posthoc_cell("random12", "seen_mt"), posthoc_cell("random12", "unseen_mt"), posthoc_cell("random12", "unseen_st"), posthoc_cell("random12", "unseen_forget"), posthoc_cell("random12", "overall_mt")],
        ["BADIT-GG assignment", 15, posthoc_cell("gg", "seen_mt"), posthoc_cell("gg", "unseen_mt"), posthoc_cell("gg", "unseen_st"), posthoc_cell("gg", "unseen_forget"), posthoc_cell("gg", "overall_mt")],
    ]
    ws = new_sheet(wb, "Table_VII", "PDF p.12: Category-held-out discovery transfer. Cells update from locally returned M6 folds.")
    add_table(ws, ["Discovery protocol", "#Calib. tasks", "Seen MT↑", "Unseen MT↑", "Unseen ST↑", "Unseen forget↓", "Overall MT↑"], rows)
    m6_seq_done = posthoc["table_vii"]["four_category"]["sequential_complete"] and posthoc["table_vii"]["random12"]["sequential_complete"]
    audits.append(["Table VII", 12, "complete" if m6_seq_done else "partial", "All15/GG complete; M6 mixed complete" + ("; M6 sequential complete" if m6_seq_done else ""), "Waiting only for M6 sequential return." if not m6_seq_done else "All cells recovered.", "M0_M1+M6+TPAMI_POSTHOC"])

    # Table VIII.
    rows = []
    for label, key in [("Random balanced", "random_balanced"), ("Contiguous SVD", "contiguous"), ("BADIT-GG (DOG)", None), ("BADIT-TF", "tf")]:
        if key is None:
            x = m7_dog["macro"]
            rows.append([label, f(x["task_split_ari"]["mean"]), f(x["seed_ari"]["mean"]), f(x["batch_ari"]["mean"]), f(x["cross_layer_cosine"]["mean"])])
        else:
            x = m7["macro"][key]
            rows.append([label, f(x["task_split_ari"]["mean"]), f(x["seed_ari"]["mean"]), f(x["batch_ari"]["mean"]), f(x["cross_layer_cosine"]["mean"])])
    ws = new_sheet(wb, "Table_VIII", "PDF p.12: Partition stability and cross-layer correspondence; bootstrap CIs are not present in the aggregate.")
    add_table(ws, ["Method", "Task-split ARI↑", "Seed ARI↑", "Batch ARI↑", "Cross-layer cosine↑"], rows)
    audits.append(["Table VIII", 12, "complete", "All four method rows", "Point estimates are macro means over three small models; per-model dispersion is retained in M7 artifacts.", "M7+M7_DOG"])

    # Table IX. Do not confuse top1_mass with Top-1 task performance.
    macro = m8["macro"]
    rows = [[method, *([NEEDED] * 7)] for method in ("Random balanced", "Contiguous SVD", "BADIT-GG (DOG)")]
    topk = posthoc["table_ix_tf_topk"]["macro"]
    rows.append([
        "BADIT-TF", f(macro["target_deletion_increase"]["mean"]), f(macro["control_deletion_increase"]["mean"]),
        f(macro["specificity_gap"]["mean"]), f(macro["sufficiency"]["mean"]), f(topk["top1_score"]["mean"]), f(topk["top4_score"]["mean"]),
        f(macro["composition_gain"]["mean"]),
    ])
    ws = new_sheet(wb, "Table_IX", "PDF p.13: Held-out intervention specificity and composition. Top-1/Top-4 task performance remains unavailable.")
    add_table(ws, ["Method", "Target del.↑", "Control del.↓", "Spec. gap↑", "Sufficiency↑", "Top-1↑", "Top-4↑", "Comp. gain↑"], rows)
    audits.append(["Table IX", 13, "partial", "All seven BADIT-TF metrics", "Three comparison rows remain protocol-ambiguous and were not fabricated.", "M8+TPAMI_POSTHOC"])

    # Table XI: paired loop-only TF/LoRAMoE timing plus frozen calibration/assignment.
    by_name = {item["model"]: item for item in table_xi["models"].values()}
    rows = [[model, f(by_name[model]["badit_tf_total_over_loramoe"], 6)] for model in MODEL_ORDER]
    rows.append(["Average", f(table_xi["average"], 6)])
    ws = new_sheet(wb, "Table_XI", "PDF p.14: Blank BADIT-TF total-cost column only; paper baseline columns are already populated.")
    add_table(ws, ["Model", "BADIT-TF total / LoRAMoE"], rows)
    audits.append(["Table XI", 14, "complete", "7/7", "Paired SFT timing arms use the same frozen seed-1 token budget; test was not loaded.", "M9+TABLE_XI"])

    # Table XII, with explicitly partial train/parameter ratios where recoverable.
    rows = []
    for model in MODEL_ORDER:
        groups = [m9["by_model_setting"].get(f"{model}.{setting}", {}) for setting in ("mixed", "sequential")]
        train_groups = [x.get("train_time_tf_over_gg", {"mean": None, "n": 0}) for x in groups]
        param_groups = [x.get("trainable_params_tf_over_gg", {"mean": None, "n": 0}) for x in groups]
        train, train_n = weighted_group_mean(train_groups)
        params, params_n = weighted_group_mean(param_groups)
        train_cell = f(train, 6) if train_n else NEEDED
        param_cell = f(posthoc["table_xii_trainable_params"][model]["tf_over_gg"], 6)
        assignment_cell = f(posthoc["table_xii_assignment_timing"][model]["assignment_solve_seconds"], 6)
        calibration = posthoc["table_xii_calibration_timing"][model]
        throughput_cell = f(posthoc["table_xii_throughput"][model]["tf_over_gg"], 6)
        rows.append([model, f(calibration["calibration_gpu_hours"], 6), assignment_cell, f(calibration["peak_memory_allocated_gb"], 6), train_cell, throughput_cell, param_cell])
    ws = new_sheet(wb, "Table_XII", "PDF p.14: Detailed BADIT-TF costs. Train/parameter ratios are partial recoveries; no imputation.")
    add_table(ws, ["Model", "Gate/Fisher calib. (GPU-hours)↓", "Assignment solve (seconds)↓", "Peak memory (GB)↓", "Train time (ratio)↓", "Infer. throughput (ratio)↑", "Trainable params (ratio)"], rows)
    audits.append(["Table XII", 14, "complete", "Calibration/memory/assignment/throughput/parameter 6/6; train time available-only aggregation", "All cells are filled; train ratio retains explicit incomplete paired-history coverage.", "M9+TPAMI_POSTHOC"])

    # Table XIX.
    a0_by_model = {x["model"]: x for x in d["A0"]["rows"]}
    rows = []
    for model in MODEL_ORDER:
        x = a0_by_model[model]
        rows.append([model, x["assignment_probes_per_task"], x["fisher_probes_per_task"], x["fidelity_probes_per_task"], x["epsilon_f"], x["solver_restarts"], x["max_iterations"]])
    ws = new_sheet(wb, "Table_XIX", "PDF p.29: BADIT-TF calibration and solver configuration.")
    add_table(ws, ["Model", "Assign./task", "Fisher/task", "Fidelity/task", "epsilon_f", "Restarts", "Max iterations"], rows)
    audits.append(["Table XIX", 29, "complete", "36/36", "All six model rows available.", "A0"])

    # Table XX.
    rows = []
    for model in MODEL_ORDER:
        x = posthoc["table_xx"][model]
        rows.append([model, f(x["iterations_mean"]), f(x["convergence_rate"]), f(x["best_regret_mean"]), f(x["restart_spread_cv_mean"])])
    ws = new_sheet(wb, "Table_XX", "PDF p.29: TF assignment-solver convergence over injected layers.")
    add_table(ws, ["Model", "Iter.↓", "Conv. rate↑", "Best regret↓", "Restart spread↓"], rows)
    audits.append(["Table XX", 29, "complete", "24/24", "Large-model rows use their own frozen TF solver audits.", "A1+TPAMI_POSTHOC"])

    # Table XXI.
    factor_labels = {"assignment_probes_per_task": "Assign. probes/task", "fisher_probes_per_task": "Fisher samples/task", "epsilon_f": "epsilon_f", "solver_restarts": "Restarts"}
    rows = [[factor_labels[x["factor"]], x["value"], f(x["heldout_regret"]), f(x["stability_ari"]), f(x["fidelity_spearman"]), f(x["calibration_time_seconds"])] for x in posthoc["table_xxi"]]
    ws = new_sheet(wb, "Table_XXI", "PDF p.30: Qwen3-4B one-factor calibration sensitivity; no formal A2 aggregate exists.")
    add_table(ws, ["Factor", "Value", "Held-out regret↓", "Stability ARI↑", "Fidelity↑", "Calib. time↓"], rows)
    audits.append(["Table XXI", 30, "complete", "64/64 result cells", "Timing is cached-profile construction plus exact assignment solve, median of five exact replays.", "H0+TPAMI_POSTHOC"])

    ws = new_sheet(wb, "Mapping_Audit", "每张 PDF 空表的填充状态、缺口和实验来源。")
    add_table(ws, ["Paper table", "PDF page", "Status", "Currently fillable", "Missing reason / caveat", "Evidence"], audits)

    artifact_rows = [["TPAMI PDF", str(pdf), pdf.stat().st_size, sha256(pdf)]]
    for label, path in paths.items():
        artifact_rows.append([label, str(path.relative_to(root)), path.stat().st_size, sha256(path)])
    for path in sorted(set(m6_results)):
        artifact_rows.append([path.parent.name, str(path.relative_to(root)), path.stat().st_size, sha256(path)])
    ws = new_sheet(wb, "Artifact_Index", "PDF 模板与所有数值来源的 SHA256。")
    add_table(ws, ["Artifact", "Path", "Bytes", "SHA256"], artifact_rows)

    # Cell comments make derivations visible without changing the paper schemas.
    wb["Table_IV"]["E5"].comment = Comment("Rel. err. uses the macro mean of per-model median relative error from M3.", "Codex")
    for cell in ("B5", "C5", "B10", "C10"):
        wb["Table_VI"][cell].comment = Comment("Three-small-model reduction of M2 to match the Table VI scope.", "Codex")
    for row in range(5, 11):
        wb["Table_XII"].cell(row, 2).comment = Comment("Selected-count assignment/Fisher collection only; model/tokenizer/bank load and assignment solve excluded; 8xL20Z GPU-hours.", "Codex")
        wb["Table_XII"].cell(row, 3).comment = Comment("Median of five exact deterministic CPU replays; every layer assignment matched the immutable frozen assignment.", "Codex")
        wb["Table_XII"].cell(row, 4).comment = Comment("Maximum rank-local torch.cuda.max_memory_allocated during the same selected-count calibration replay.", "Codex")
        wb["Table_XII"].cell(row, 5).comment = Comment("Partial matched-run recovery; coverage is documented in Mapping_Audit and M9.", "Codex")
        wb["Table_XII"].cell(row, 6).comment = Comment("Fixed-load aggregate 8-GPU generation throughput: 15 frozen calibration prompts, five repeats, exactly 32 generated tokens, rank-order counterbalanced; TF/GG ratio.", "Codex")
        wb["Table_XII"].cell(row, 7).comment = Comment("Partial matched-run recovery; no missing cell was imputed.", "Codex")

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".xlsx", dir=output.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        wb.save(tmp)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    check = load_workbook(output, read_only=True, data_only=False)
    sheets = {ws.title: {"rows": ws.max_row, "columns": ws.max_column} for ws in check.worksheets}
    check.close()
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "pdf": str(pdf),
        "pdf_sha256": sha256(pdf),
        "workbook": str(output.relative_to(root)),
        "workbook_sha256": sha256(output),
        "workbook_bytes": output.stat().st_size,
        "paper_blank_tables": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "XI", "XII", "XIX", "XX", "XXI"],
        "sheets": sheets,
        "sources": {label: {"path": str(path.relative_to(root)), "sha256": sha256(path)} for label, path in paths.items()},
        "missing_marker": NEEDED,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "experiments/results/BADIT_TF_TPAMI_EMPTY_TABLES.xlsx"
    manifest = make_workbook(root, args.pdf.resolve(), output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"workbook": str(output), "sha256": manifest["workbook_sha256"], "sheets": len(manifest["sheets"]), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
