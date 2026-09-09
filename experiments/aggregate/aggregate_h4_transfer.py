#!/usr/bin/env python3
"""Fail-closed validation-only selection lock for the H4 large-model transfer grid.

The H3 lock supplies three candidates per family/setting, except the explicitly
authorized Qwen mixed top-four amendment.  H4 executes matched TF/GG once per
candidate on each corresponding large model.  A Qwen GG source attempt is
    immutable failed evidence; its isolated r5 recovery is the sole eligible
replacement for that one logical cell.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(".")
LOCAL = ROOT / "experiments/raw_results/h4_returned_results"
OUT = ROOT / "experiments/raw_results/h4_confirmation_aggregate/selection_lock.json"
MANIFESTS = (
    ROOT / "experiments/materials/h4_qwen3_8b_dispatch_manifest.json",
    ROOT / "experiments/materials/h4_llama3_8b_dispatch_manifest.json",
    ROOT / "experiments/materials/h4_gemma2_9b_dispatch_manifest.json",
)
RECOVERY = {
    "h4_qwen3_8b_sequential_h2_06_gg_seed1": "h4_qwen3_8b_sequential_h2_06_gg_seed1_r5",
}
LLAMA_RECOVERY_MANIFEST = ROOT / "experiments/materials/h4_llama3_8b_assignment_hash_recovery_r1.json"
GPU_HOURS_AUDIT = ROOT / "experiments/raw_results/h4_gpu_hours_scheduler_history/audit.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def recovery_metadata() -> dict[str, dict[str, Any]]:
    """Load the one grid-wide Llama preflight recovery without mutating history."""
    if not LLAMA_RECOVERY_MANIFEST.is_file():
        return {}
    payload = load(LLAMA_RECOVERY_MANIFEST)
    require(payload.get("event") == "H4_LLAMA_ASSIGNMENT_HASH_RECOVERY", "invalid Llama recovery manifest")
    rows = payload.get("recovery_trials", [])
    require(len(rows) == 12, "wrong Llama recovery cell count")
    mapped = {str(row["run_id"]): row for row in rows}
    require(len(mapped) == len(rows), "duplicate Llama recovery source")
    for logical_id, row in mapped.items():
        require(str(row.get("recovery_run_id", "")) == f"{logical_id}_r1", f"bad recovery identity: {logical_id}")
        require(isinstance(row.get("recovery_config_sha256"), str), f"missing recovery config hash: {logical_id}")
    return mapped


def duration_seconds(value: object) -> int:
    require(isinstance(value, str) and re.fullmatch(r"\d+:\d{2}:\d{2}", value) is not None, f"invalid scheduler duration: {value!r}")
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def scheduler_gpu_hours(record: dict[str, Any], audit: dict[str, Any]) -> float:
    run_id = str(record["effective_run_id"])
    item = audit.get("evidence", {}).get(run_id, {})
    require(item.get("status") == "RECOVERED_PM_HISTORY", f"missing scheduler GPU-hours evidence: {run_id}")
    require(item.get("scheduler_status") == "success" and item.get("scheduler_exit_code") == 0, f"non-success scheduler record: {run_id}")
    require(item.get("gpu_count") == 8, f"GPU-count drift in cost evidence: {run_id}")
    return duration_seconds(item.get("scheduler_duration")) * 8 / 3600


def result_for(logical_id: str, recoveries: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any], Path]:
    effective_id = RECOVERY.get(logical_id, str(recoveries.get(logical_id, {}).get("recovery_run_id", logical_id)))
    path = LOCAL / effective_id / "result.json"
    result = load(path)
    require(result.get("run_id") == effective_id, f"result identity mismatch: {effective_id}")
    return effective_id, result, path


def validate_trial(manifest: dict[str, Any], trial: dict[str, Any], llama_recoveries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    logical_id = str(trial["run_id"])
    recovery = llama_recoveries.get(logical_id)
    effective_id, result, result_path = result_for(logical_id, llama_recoveries)
    config_path = ROOT / str(recovery["recovery_config_path"] if recovery else trial["config_path"])
    expected_config_sha = recovery["recovery_config_sha256"] if recovery else trial["config_sha256"]
    require(sha(config_path) == expected_config_sha, f"config hash drift: {logical_id}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(result.get("status") == "complete", f"non-complete H4 result: {effective_id}")
    assertions = result.get("metrics", {}).get("assertions", {})
    require(bool(assertions) and all(value is True for value in assertions.values()), f"assertion failure: {effective_id}")
    protocol = result.get("protocol", {})
    require(protocol.get("evaluation_role") == "tune_validation", f"non-validation result: {effective_id}")
    require(protocol.get("official_test_loaded", False) is False, f"official test loaded: {effective_id}")
    require(protocol.get("official_test_used_for_selection", protocol.get("final_test_used_for_selection", False)) is False, f"test used: {effective_id}")
    require(result.get("config_sha256") == sha(config_path), f"result/config SHA mismatch: {effective_id}")
    require(config.get("h4_validation_only") is True, f"missing validation fence: {logical_id}")
    require(config.get("official_test_used_for_selection") is False, f"config test fence drift: {logical_id}")
    require(config.get("h4_h3_selection_lock_sha256") == manifest["selection_lock_sha256"], f"H3 lock drift: {logical_id}")
    expected_assignment_sha = llama_recoveries and recovery and load(LLAMA_RECOVERY_MANIFEST)["actual_assignment_sha256"] or manifest["assignment_sha256"]
    require(config.get("h4_model_native_assignment_sha256") == expected_assignment_sha, f"assignment drift: {logical_id}")
    # Qwen R5 deliberately keeps the immutable, logical-cell config (and its
    # SHA) while emitting a distinct effective result ID.  Llama R1 recovery
    # configs instead carry their effective IDs.  Validate both provenance
    # contracts explicitly rather than requiring those namespaces to match.
    expected_config_run_id = logical_id if logical_id in RECOVERY else effective_id
    require(config.get("run_id") == expected_config_run_id, f"config run ID drift: {logical_id}")
    if logical_id in RECOVERY:
        require(result.get("recovery_source_run_id") is None, f"Qwen R5 reused a source run: {effective_id}")
    # Mixed writers record a top-level variant; the sequential aggregator's
    # stable schema records it only in protocol.  When present, both must
    # agree, while protocol is the cross-setting authoritative field.
    if result.get("variant") is not None:
        require(result.get("variant") == trial["variant"], f"result variant drift: {effective_id}")
    require(protocol.get("assignment_method") == trial["variant"], f"protocol variant drift: {effective_id}")
    metric: dict[str, float]
    if trial["setting"] == "mixed":
        primary = result.get("metrics", {}).get("rouge", {}).get("macro", {}).get("rougeL")
        require(isinstance(primary, (int, float)), f"missing mixed macro score: {effective_id}")
        metric = {"primary": float(primary)}
    else:
        continual = result.get("metrics", {}).get("continual", {})
        values = {key: continual.get(key) for key in ("continual_score", "forget_rate", "backward")}
        require(all(isinstance(value, (int, float)) for value in values.values()), f"missing sequential metrics: {effective_id}")
        metric = {"primary": float(values["continual_score"]), "forget_rate": float(values["forget_rate"]), "backward": float(values["backward"])}
    return {
        "logical_run_id": logical_id,
        "effective_run_id": effective_id,
        "recovery_used": logical_id != effective_id,
        "source_trial": trial["source_trial"],
        "variant": trial["variant"],
        "setting": trial["setting"],
        "config_path": str(config_path),
        "config_sha256": sha(config_path),
        "result_path": str(result_path),
        "result_sha256": sha(result_path),
        **metric,
    }


def rank(rows: list[dict[str, Any]], variant: str, setting: str) -> None:
    candidates = list(rows)
    if setting == "mixed":
        ordered = sorted(candidates, key=lambda row: (-row[variant]["primary"], row["source_trial"]))
    else:
        best = max(row[variant]["primary"] for row in candidates)
        close = [row for row in candidates if best - row[variant]["primary"] <= 0.1]
        distant = [row for row in candidates if row not in close]
        ordered = sorted(close, key=lambda row: (row[variant]["forget_rate"], -row[variant]["backward"], row["source_trial"]))
        ordered += sorted(distant, key=lambda row: (-row[variant]["primary"], row["source_trial"]))
    for index, row in enumerate(ordered, 1):
        row[variant]["validation_rank"] = index


def main() -> None:
    manifests = [load(path) for path in MANIFESTS]
    require(sum(len(item.get("trials", [])) for item in manifests) == 38, "H4 dispatch must contain 38 logical cells")
    require(all(item.get("test_closed") is True for item in manifests), "dispatch test closure drift")
    h3_hashes = {item.get("selection_lock_sha256") for item in manifests}
    require(len(h3_hashes) == 1 and None not in h3_hashes, "inconsistent H3 parent lock")
    llama_recoveries = recovery_metadata()
    llama_logical = {str(row["run_id"]) for manifest in manifests if manifest.get("model") == "Llama3-8B" for row in manifest["trials"]}
    require(set(llama_recoveries) == llama_logical, "Llama recovery does not cover exactly the original grid")
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for manifest in manifests:
        model = str(manifest["model"])
        for trial in manifest["trials"]:
            record = validate_trial(manifest, trial, llama_recoveries)
            records.append(record)
            candidate = groups.setdefault((model, record["setting"]), {}).setdefault(record["source_trial"], {"source_trial": record["source_trial"]})
            require(record["variant"] not in candidate, f"duplicate logical variant: {record['logical_run_id']}")
            candidate[record["variant"]] = record
    require(len(records) == 38, "wrong accepted H4 record count")
    selected: dict[str, Any] = {}
    ranked: dict[str, list[dict[str, Any]]] = {}
    gpu_hours_audit_sha256: str | None = None
    for (model, setting), candidates in sorted(groups.items()):
        expected = 4 if model == "Qwen3-8B" and setting == "mixed" else 3
        require(len(candidates) == expected, f"wrong candidate count: {model}/{setting}")
        rows = []
        for source_trial, item in candidates.items():
            require(set(item) == {"source_trial", "tf", "gg"}, f"incomplete matched pair: {model}/{setting}/{source_trial}")
            rows.append(item)
        rank(rows, "tf", setting); rank(rows, "gg", setting)
        for row in rows:
            ranks = [row[variant]["validation_rank"] for variant in ("tf", "gg")]
            row["selection"] = {"mean_validation_rank": statistics.mean(ranks), "worst_validation_rank": max(ranks)}
        rank_key = lambda row: (row["selection"]["mean_validation_rank"], row["selection"]["worst_validation_rank"])
        rows.sort(key=lambda row: (*rank_key(row), row["source_trial"]))
        if len(rows) > 1 and rank_key(rows[0]) == rank_key(rows[1]):
            audit = load(GPU_HOURS_AUDIT)
            require(audit.get("event") == "H4_GPU_HOURS_SCHEDULER_HISTORY_AUDIT", "invalid H4 GPU-hours audit")
            require(audit.get("selection_performed") is False, "GPU-hours audit performed selection")
            gpu_hours_audit_sha256 = sha(GPU_HOURS_AUDIT)
            for row in rows:
                row["selection"]["scheduler_gpu_hours"] = sum(scheduler_gpu_hours(row[variant], audit) for variant in ("tf", "gg"))
            rows.sort(key=lambda row: (*rank_key(row), row["selection"]["scheduler_gpu_hours"], row["source_trial"]))
            if len(rows) > 1 and (*rank_key(rows[0]), rows[0]["selection"]["scheduler_gpu_hours"]) == (*rank_key(rows[1]), rows[1]["selection"]["scheduler_gpu_hours"]):
                raise ValueError(f"unresolved H4 rank/GPU-hours tie: {model}/{setting}")
        winner = rows[0]
        winner["selection"]["shared_config_rank"] = 1
        key = f"{model}/{setting}"
        selected[key] = {"source_trial": winner["source_trial"], "tf": winner["tf"], "gg": winner["gg"], "selection": winner["selection"]}
        ranked[key] = rows
    output = {
        "schema_version": 1,
        "event": "H4_LARGE_MODEL_TRANSFER_VALIDATION_SELECTION_LOCK",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_logical_cells": 38,
        "accepted_logical_cells": len(records),
        "source_h3_selection_lock_sha256": next(iter(h3_hashes)),
        "selection_metric_source": "tune_validation",
        "official_test_loaded": False,
        "official_test_used_for_selection": False,
        "gpu_hours_audit_path": str(GPU_HOURS_AUDIT) if gpu_hours_audit_sha256 else None,
        "gpu_hours_audit_sha256": gpu_hours_audit_sha256,
        "recovery_mapping": {**RECOVERY, **{key: value["recovery_run_id"] for key, value in llama_recoveries.items()}},
        "selection_rule": "Rank TF and matched GG separately; mixed maximizes macro ROUGE-L; sequential maximizes continual score, with <=0.1 tie resolved by lower Forget then higher Backward; minimize mean rank, then worst rank, then preregistered GPU-hours.",
        "selected": selected,
        "ranked_candidates": ranked,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(OUT), "sha256": sha(OUT), "groups": len(selected), "records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
