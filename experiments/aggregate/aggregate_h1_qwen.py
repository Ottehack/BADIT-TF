#!/usr/bin/env python3
"""Audit and lock the completed Qwen3-4B H1 one-factor screen.

This script intentionally reads only the frozen tune-validation result files.
It refuses to emit a lock if any of the 34 preregistered Qwen rows is missing,
failed, protocol-invalid, or inconsistent with its frozen config SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_run_id(grid_run_id: str) -> str:
    return grid_run_id.replace("h1-", "h1_") + "_remote_l20z"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_row(root: Path, row: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    run_id = result_run_id(row["run_id"])
    result_path = raw_root / run_id / "result.json"
    require(result_path.is_file(), f"missing result: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))

    require(result.get("run_id") == run_id, f"run_id mismatch: {run_id}")
    require(result.get("status") == "complete", f"non-complete result: {run_id}")
    require(
        result.get("config_sha256") == row["config_sha256"],
        f"grid/result config SHA mismatch: {run_id}",
    )
    config_path = root / row["config_path"]
    require(config_path.is_file(), f"missing config: {config_path}")
    require(
        sha256_file(config_path) == row["config_sha256"],
        f"config file SHA mismatch: {run_id}",
    )

    protocol = result.get("protocol", {})
    require(
        protocol.get("evaluation_role") == "tune_validation",
        f"non-validation evaluation role: {run_id}",
    )
    official_loaded = protocol.get("official_test_loaded")
    require(
        official_loaded is not True,
        f"official test was loaded: {run_id}",
    )
    official_used = protocol.get("official_test_used_for_selection")
    final_used = protocol.get("final_test_used_for_selection")
    require(
        official_used is not True
        and final_used is not True
        and (official_used is False or final_used is False),
        f"test-selection audit missing or failed: {run_id}",
    )
    if official_loaded is None:
        require(
            row["setting"] == "sequential"
            and protocol.get("formal_budget") is True
            and official_used is False,
            f"official-test load evidence missing: {run_id}",
        )
    assertions = result.get("metrics", {}).get("assertions", {})
    require(bool(assertions), f"assertions missing: {run_id}")
    failed_assertions = sorted(key for key, value in assertions.items() if value is not True)
    require(not failed_assertions, f"failed assertions in {run_id}: {failed_assertions}")

    metrics = result["metrics"]
    audited = {
        "trial_id": row["trial_id"],
        "run_id": run_id,
        "setting": row["setting"],
        "changed_axis": row["changed_axis"],
        "learning_rate": row["learning_rate"],
        "router_lr_multiplier": row["router_lr_multiplier"],
        "warmup_ratio": row["warmup_ratio"],
        "lora_dropout": row["lora_dropout"],
        "weight_decay": row["weight_decay"],
        "config_path": row["config_path"],
        "config_sha256": row["config_sha256"],
        "result_path": str(result_path.relative_to(root)),
        "result_sha256": sha256_file(result_path),
        "evaluation_role": protocol["evaluation_role"],
        "official_test_loaded": False,
        "official_test_loaded_evidence": (
            "explicit_protocol_field"
            if official_loaded is False
            else "validation_only_sequential_evaluator_path"
        ),
        "all_assertions_passed": True,
    }
    if row["setting"] == "mixed":
        macro = metrics.get("rouge", {}).get("macro", {})
        require(isinstance(macro.get("rougeL"), (int, float)), f"mixed score missing: {run_id}")
        audited.update(
            {
                "primary_metric": "macro_task_rougeL",
                "primary_score": macro["rougeL"],
                "macro_exact_match": macro.get("exact_match"),
                "macro_rouge1": macro.get("rouge1"),
            }
        )
    else:
        continual = metrics.get("continual", {})
        for key in ("continual_score", "forget_rate", "backward"):
            require(
                isinstance(continual.get(key), (int, float)),
                f"sequential metric {key} missing: {run_id}",
            )
        audited.update(
            {
                "primary_metric": "continual_score",
                "primary_score": continual["continual_score"],
                "forget_rate": continual["forget_rate"],
                "backward": continual["backward"],
                "forward": continual.get("forward"),
            }
        )
    return audited


def choose(rows: list[dict[str, Any]], setting: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = sorted(rows, key=lambda item: (-item["primary_score"], item["trial_id"]))
    for rank, row in enumerate(ranked, start=1):
        row["primary_rank"] = rank
    if setting == "mixed":
        return ranked[0], ranked

    best_score = ranked[0]["primary_score"]
    tie_band = [row for row in ranked if best_score - row["primary_score"] <= 0.1]
    selected = min(
        tie_band,
        key=lambda item: (item["forget_rate"], -item["backward"], item["trial_id"]),
    )
    return selected, ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/configs/h1_one_factor_grid.json")
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("experiments/raw_results")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/raw_results/h1_qwen3_4b_one_factor_aggregate/result.json"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(
            "experiments/raw_results/h1_qwen3_4b_one_factor_aggregate/selection_lock.json"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    grid_path = root / args.grid
    raw_root = root / args.raw_root
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    qwen_rows = [row for row in grid["trial_rows"] if row["model"] == "Qwen3-4B"]
    require(len(qwen_rows) == 34, f"expected 34 Qwen rows, found {len(qwen_rows)}")
    audited = [audit_row(root, row, raw_root) for row in qwen_rows]

    by_setting = {}
    for setting in ("mixed", "sequential"):
        setting_rows = [row for row in audited if row["setting"] == setting]
        require(len(setting_rows) == 17, f"expected 17 {setting} rows")
        selected, ranked = choose(setting_rows, setting)
        by_setting[setting] = {"selected": selected, "ranked_rows": ranked}

    timestamp = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": 1,
        "experiment_id": "H1-ONE-FACTOR-QWEN3-4B",
        "status": "complete_locked",
        "generated_at": timestamp,
        "grid_path": str(args.grid),
        "grid_sha256": sha256_file(grid_path),
        "completed_rows": 34,
        "expected_rows": 34,
        "selection_metric_source": "tune_validation",
        "official_test_loaded": False,
        "test_used_for_selection": False,
        "selection_rule": {
            "mixed": "maximize macro task ROUGE-L; trial_id ascending only for exact ties",
            "sequential": (
                "maximize continual score; among trials within 0.1 point of the maximum, "
                "minimize Forget Rate, then maximize Backward, then trial_id ascending"
            ),
        },
    }
    output = {**common, "settings": by_setting, "audited_rows": audited}
    lock = {
        **common,
        "selected": {
            setting: {
                key: value
                for key, value in payload["selected"].items()
                if key
                in {
                    "trial_id",
                    "run_id",
                    "primary_metric",
                    "primary_score",
                    "forget_rate",
                    "backward",
                    "config_path",
                    "config_sha256",
                    "result_path",
                    "result_sha256",
                }
            }
            for setting, payload in by_setting.items()
        },
    }

    output_path = root / args.output
    lock_path = root / args.lock
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "lock": str(args.lock), "selected": lock["selected"]}, indent=2))


if __name__ == "__main__":
    main()
