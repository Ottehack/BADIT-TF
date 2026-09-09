#!/usr/bin/env python3
"""Fail-closed aggregation and selection lock for the full H3 confirmation grid.

H2 supplies the TF seed-1 observation of each locked candidate.  H3 adds
TF seeds 2/3 and matched GG seeds 1/2/3, so a completed lock always audits
30 candidates x 6 validation observations = 180 source result files.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openpyxl import load_workbook


ROOT = Path(".")
LOCK_PATH = ROOT / "experiments/raw_results/h2_joint_search_aggregate/selection_lock.json"
AMENDMENT_PATH = ROOT / "experiments/materials/h3_qwen_mixed_top4_protocol_amendment.json"
RESULT_ROOT = ROOT / "experiments/raw_results"
EXPECTED_VARIANTS = {"tf": (2, 3), "gg": (1, 2, 3)}
FROZEN_FIELDS = (
    "model_name", "setting", "trial_id", "data_root", "tuning_split_manifest",
    "tf_assignment_path", "tf_assignment_epsilon", "initial_bank_checkpoint",
    "initial_bank_checkpoint_sha256", "world_size", "max_sequence_length",
    "max_target_length", "max_new_tokens", "torch_dtype", "target_modules",
    "num_experts", "rank", "lora_alpha", "apply_lora_dropout", "top_k",
    "dense_steps", "router_mode", "router_bias", "initialization_method",
    "svd_method", "residual_implementation", "route_pooling_scope", "adam_betas",
    "adam_epsilon", "gradient_clipping", "loss_explosion_factor", "setting",
    "epochs", "order_manifest", "official_test_manifest", "learning_rate",
    "router_lr_multiplier", "warmup_ratio", "lora_dropout", "weight_decay",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def qwen_mixed_top4_amendment() -> dict[str, Any]:
    """Validate the narrowly scoped owner-authorized H3 tie amendment."""
    amendment = load_json(AMENDMENT_PATH)
    require(amendment.get("event") == "H3_GPU_HOURS_UNRECOVERABLE_TIE_H4_TOP4_AMENDMENT", "wrong H3 tie amendment")
    require(amendment.get("status") == "authorized", "H3 tie amendment is not authorized")
    scope = amendment.get("scope", {})
    require(scope.get("model_setting_group") == "Qwen3-4B/mixed", "H3 tie amendment has wrong scope")
    require(
        scope.get("tied_h2_candidate_run_ids") == [
            "h2_qwen3_4b_mixed_h2_01_seed1_r1_r2",
            "h2_qwen3_4b_mixed_h2_17_seed1",
        ],
        "H3 tie amendment has wrong candidates",
    )
    source = amendment.get("source_evidence", {})
    require(source.get("h2_validation_only_lock_sha256") == digest(LOCK_PATH), "H3 tie amendment H2-lock hash drift")
    require(
        source.get("tie_audit_sha256") == digest(ROOT / "experiments/raw_results/H3_SELECTION_TIE_AUDIT.md"),
        "H3 tie amendment tie-audit hash drift",
    )
    require(
        source.get("scheduler_recovery_audit_sha256")
        == digest(ROOT / "experiments/raw_results/h3_gpu_hours_scheduler_recovery/audit.json"),
        "H3 tie amendment scheduler-audit hash drift",
    )
    return amendment


def test_closed(protocol: dict[str, Any], run_id: str) -> None:
    require(protocol.get("evaluation_role") == "tune_validation", f"non-validation result: {run_id}")
    require(protocol.get("official_test_loaded", False) is False, f"official test loaded: {run_id}")
    require(
        protocol.get("official_test_used_for_selection", protocol.get("final_test_used_for_selection", False)) is False,
        f"official test used for selection: {run_id}",
    )


def score_record(
    *, result: dict[str, Any], result_path: Path, config: dict[str, Any],
    config_path: Path, variant: str, seed: int, expected_h2: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(result.get("run_id", ""))
    require(result.get("status") == "complete", f"non-complete result: {run_id}")
    assertions = result.get("metrics", {}).get("assertions", {})
    require(bool(assertions) and all(value is True for value in assertions.values()), f"assertion failure: {run_id}")
    test_closed(result.get("protocol", {}), run_id)
    require(result.get("config_sha256") == digest(config_path), f"result/config SHA mismatch: {run_id}")
    require(config.get("model_name") == expected_h2["model_name"], f"model mismatch: {run_id}")
    require(config.get("setting") == expected_h2["setting"], f"setting mismatch: {run_id}")
    require(config.get("trial_id") == expected_h2["trial_id"], f"trial mismatch: {run_id}")
    for field in FROZEN_FIELDS:
        if field in expected_h2["config"]:
            require(config.get(field) == expected_h2["config"].get(field), f"frozen field drift {field}: {run_id}")
    if seed != 1:  # H2 is the frozen TF seed-1 source; H3 carries the variant flag.
        if variant == "gg":
            require(config.get("assignment_method") == "tf" and config.get("h3_variant") == "gg", f"GG contract drift: {run_id}")
        else:
            require(config.get("assignment_method") == "tf" and config.get("h3_variant") in {None, "tf"}, f"TF contract drift: {run_id}")
    metrics = result["metrics"]
    if expected_h2["setting"] == "mixed":
        primary = metrics.get("rouge", {}).get("macro", {}).get("rougeL")
        require(isinstance(primary, (int, float)), f"missing mixed primary: {run_id}")
        extra: dict[str, float] = {}
    else:
        continual = metrics.get("continual", {})
        primary = continual.get("continual_score")
        require(isinstance(primary, (int, float)), f"missing sequential primary: {run_id}")
        extra = {}
        for name in ("forget_rate", "backward"):
            value = continual.get(name)
            require(isinstance(value, (int, float)), f"missing sequential {name}: {run_id}")
            extra[name] = float(value)
    return {
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "primary": float(primary),
        **extra,
        "config_path": str(config_path),
        "config_sha256": digest(config_path),
        "result_path": str(result_path),
        "result_sha256": digest(result_path),
        "code_commit": result.get("code_commit", result.get("aggregation_code_commit")),
    }


def source_h2_record(item: dict[str, Any]) -> dict[str, Any]:
    config_path = ROOT / item["config_path"]
    result_path = ROOT / item["result_path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = load_json(result_path)
    require(digest(config_path) == item["config_sha256"], f"H2 config hash drift: {item['run_id']}")
    require(digest(result_path) == item["result_sha256"], f"H2 result hash drift: {item['run_id']}")
    expected = {"model_name": config["model_name"], "setting": config["setting"], "trial_id": config["trial_id"], "config": config}
    record = score_record(
        result=result, result_path=result_path, config=config, config_path=config_path,
        variant="tf", seed=1, expected_h2=expected,
    )
    require(record["run_id"] == item["run_id"], f"H2 run identity drift: {item['run_id']}")
    return record


def registry_h3_rows() -> list[dict[str, Any]]:
    book = load_workbook(ROOT / "experiments/experiment_registry.xlsx", read_only=True, data_only=True)
    sheet = book["HPARAM"]
    headers = {cell.value: index for index, cell in enumerate(sheet[1])}
    rows = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        run_id = str(row[headers["run_id"]] or "")
        if run_id.startswith("h3_") and row[headers["status"]] == "complete":
            rows.append({name: row[index] for name, index in headers.items()})
    return rows


def h3_records(expected: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        config_path = ROOT / str(row["config_path"])
        result_path = ROOT / str(row["raw_result_path"])
        if not config_path.is_file() or not result_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("h2_candidate_run_id") != expected["run_id"]:
            continue
        variant = config.get("h3_variant") or str(row.get("method") or "").lower()
        if variant not in EXPECTED_VARIANTS:
            continue
        if config.get("h3_variant") is None:
            # The first frozen TF dispatch expressed its variant only in the
            # hash-bound wrapper.  Treat it as legacy evidence only after
            # independently checking both the registry method and wrapper.
            require(variant == "tf", f"undeclared non-TF H3 variant: {row['run_id']}")
            wrapper = ROOT / str(row.get("bash_runner") or "")
            require(wrapper.is_file(), f"missing legacy TF wrapper: {row['run_id']}")
            text = wrapper.read_text(encoding="utf-8")
            require("H3_VARIANT='tf'" in text or "H3_VARIANT=tf" in text, f"legacy TF wrapper drift: {row['run_id']}")
        seed = int(config.get("seed"))
        if seed not in EXPECTED_VARIANTS[variant]:
            continue
        result = load_json(result_path)
        record = score_record(
            result=result, result_path=result_path, config=config, config_path=config_path,
            variant=variant, seed=seed, expected_h2=expected,
        )
        key = (variant, seed)
        if key in matches:
            raise ValueError(
                f"duplicate completed H3 logical cell {expected['run_id']} {key}: "
                f"{matches[key]['run_id']} vs {record['run_id']}"
            )
        matches[key] = record
    missing = [f"{variant}:seed{seed}" for variant, seeds in EXPECTED_VARIANTS.items() for seed in seeds if (variant, seed) not in matches]
    require(not missing, f"H3 confirmation incomplete for {expected['run_id']}: {missing}")
    return [matches[(variant, seed)] for variant, seeds in EXPECTED_VARIANTS.items() for seed in seeds]


def summarize(candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"candidate": candidate["h2"], "records": candidate["records"], "variants": {}}
    for variant in ("tf", "gg"):
        values = [row for row in candidate["records"] if row["variant"] == variant]
        primary = [row["primary"] for row in values]
        summary: dict[str, Any] = {
            "n": len(values), "seeds": [row["seed"] for row in values],
            "primary_mean": statistics.mean(primary),
            "primary_sample_std": statistics.stdev(primary),
        }
        if candidate["h2"]["setting"] == "sequential":
            for field in ("forget_rate", "backward"):
                numbers = [row[field] for row in values]
                summary[f"{field}_mean"] = statistics.mean(numbers)
                summary[f"{field}_sample_std"] = statistics.stdev(numbers)
        result["variants"][variant] = summary
    return result


def rank_variant(rows: list[dict[str, Any]], variant: str, setting: str) -> None:
    if setting == "mixed":
        ordered = sorted(rows, key=lambda row: (-row["variants"][variant]["primary_mean"], row["candidate"]["trial_id"]))
    else:
        best = max(row["variants"][variant]["primary_mean"] for row in rows)
        near = [row for row in rows if best - row["variants"][variant]["primary_mean"] <= 0.1]
        far = [row for row in rows if row not in near]
        ordered = sorted(near, key=lambda row: (row["variants"][variant]["forget_rate_mean"], -row["variants"][variant]["backward_mean"], row["candidate"]["trial_id"]))
        ordered += sorted(far, key=lambda row: (-row["variants"][variant]["primary_mean"], row["candidate"]["trial_id"]))
    for rank, row in enumerate(ordered, 1):
        row["variants"][variant]["validation_rank"] = rank


def main() -> None:
    lock = load_json(LOCK_PATH)
    require(lock.get("event") == "H2_VALIDATION_ONLY_TOP5_LOCK", "wrong H2 lock event")
    selected = lock.get("selected", {})
    require(len(selected) == 6 and all(len(values) == 5 for values in selected.values()), "H2 top-5 lock is incomplete")
    rows = registry_h3_rows()
    amendment = qwen_mixed_top4_amendment()
    groups: dict[str, list[dict[str, Any]]] = {}
    h4_candidate_counts: dict[str, int] = {}
    for group, items in selected.items():
        group_rows = []
        for item in items:
            h2_record = source_h2_record(item)
            h2_config = yaml.safe_load((ROOT / item["config_path"]).read_text())
            expected = {"run_id": item["run_id"], "model_name": h2_config["model_name"], "setting": group.split("/", 1)[1], "trial_id": item["trial_id"], "config": h2_config}
            records = [h2_record, *h3_records(expected, rows)]
            require(len(records) == 6, f"wrong source count: {item['run_id']}")
            group_rows.append(summarize({"h2": {**item, "model_name": expected["model_name"], "setting": expected["setting"]}, "records": records}))
        rank_variant(group_rows, "tf", group.split("/", 1)[1])
        rank_variant(group_rows, "gg", group.split("/", 1)[1])
        for row in group_rows:
            ranks = [row["variants"][variant]["validation_rank"] for variant in ("tf", "gg")]
            row["selection"] = {"mean_validation_rank": statistics.mean(ranks), "worst_validation_rank": max(ranks)}
        group_rows.sort(key=lambda row: (row["selection"]["mean_validation_rank"], row["selection"]["worst_validation_rank"], row["candidate"]["trial_id"]))
        h4_count = 3
        # GPU-hours is the final preregistered tie-break.  Formal H3 result
        # records do not expose it, so refuse a lock rather than invent it,
        # except for the one narrowly authorized Qwen mixed top-4 amendment.
        if len(group_rows) > 1 and group_rows[0]["selection"] == group_rows[1]["selection"]:
            tied = [group_rows[0]["candidate"]["run_id"], group_rows[1]["candidate"]["run_id"]]
            allowed = group == "Qwen3-4B/mixed" and tied == amendment["scope"]["tied_h2_candidate_run_ids"]
            require(allowed, f"unresolved H3 rank tie requires GPU-hours: {group}")
            h4_count = 4
            for index, row in enumerate(group_rows, 1):
                row["selection"]["shared_config_rank"] = 1 if index <= 2 else index
                row["selection"]["tie_resolution"] = {
                    "event": amendment["event"],
                    "amendment_path": str(AMENDMENT_PATH),
                    "amendment_sha256": digest(AMENDMENT_PATH),
                    "h4_candidate_count": 4,
                }
        else:
            for rank, row in enumerate(group_rows, 1):
                row["selection"]["shared_config_rank"] = rank
        h4_candidate_counts[group] = h4_count
        groups[group] = group_rows
    timestamp = datetime.now(timezone.utc).isoformat()
    output = {
        "schema_version": 1,
        "event": "H3_FULL_VALIDATION_ONLY_CONFIRMATION_AGGREGATE",
        "status": "complete_locked_after_qwen_mixed_top4_protocol_amendment",
        "generated_at": timestamp,
        "h2_selection_lock": str(LOCK_PATH),
        "h2_selection_lock_sha256": digest(LOCK_PATH),
        "h3_execution_records_expected": 150,
        "h3_execution_records_validated": 150,
        "source_records_validated": 180,
        "selection_metric_source": "tune_validation",
        "official_test_loaded": False,
        "official_test_used_for_selection": False,
        "selection_rule": "Rank TF and matched GG validation means separately; shared config minimizes mean rank, then worse rank, then GPU-hours. Sequential per-method ranking uses the 0.1-point primary band then lower Forget Rate and higher Backward. H4 receives the top three shared configs, except the explicitly authorized unresolved-GPU-hours Qwen3-4B/mixed tie, where both tied candidates are retained and H4 receives top four.",
        "protocol_amendment": {"path": str(AMENDMENT_PATH), "sha256": digest(AMENDMENT_PATH)},
        "groups": groups,
        "selected": {group: rows[:count] if count == 4 else rows[0] for group, rows in groups.items() for count in [h4_candidate_counts[group]]},
        "h4_candidate_counts": h4_candidate_counts,
        "h4_candidates": {group: rows[:h4_candidate_counts[group]] for group, rows in groups.items()},
    }
    output_path = RESULT_ROOT / "h3_confirmation_aggregate/result.json"
    lock_path = RESULT_ROOT / "h3_confirmation_aggregate/selection_lock.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (output_path, lock_path):
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_trials = {
        group: [row["candidate"]["trial_id"] for row in value]
        if isinstance(value, list)
        else [value["candidate"]["trial_id"]]
        for group, value in output["selected"].items()
    }
    print(json.dumps({"output": str(output_path), "lock": str(lock_path), "selected": selected_trials}, sort_keys=True))


if __name__ == "__main__":
    main()
