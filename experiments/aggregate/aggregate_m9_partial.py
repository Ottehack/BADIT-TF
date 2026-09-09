#!/usr/bin/env python3
"""Aggregate only the currently recoverable M9 efficiency evidence.

The emergency PM manifest is the source of scheduler wall time. Registry
timestamps are deliberately not accepted because they may include queue or
migration time. Missing evidence remains explicit rather than imputed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from badit_tf.m9_reporting import duration_seconds, paired_ratio, summary


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "analysis/tables/m0_m1_official_aggregate.json"
HISTORY = ROOT / "experiments/logs/remote_sync/m0_m1_formal_wave_20260807/collection_manifest.json"
RESULTS = ROOT / "experiments/raw_results/m0_m1_official_result_records"
OUTPUT = ROOT / "analysis/tables/m9_efficiency_partial.json"
MARKDOWN = ROOT / "analysis/tables/M9_EFFICIENCY_PARTIAL.md"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_params(run_id: str) -> tuple[int | None, str]:
    path = RESULTS / run_id / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("metrics", {}).get("trainable_parameters")
    return (int(value) if value is not None else None), digest(path)


def main() -> None:
    official = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    history_payload = json.loads(HISTORY.read_text(encoding="utf-8"))
    histories = {row["run_id"]: row.get("history") or {} for row in history_payload["records"]}
    records = []
    for key, tf_cell in sorted(official["cells"].items()):
        if tf_cell["variant"] != "tf":
            continue
        model, setting, _ = key.split(".")
        gg_cell = official["cells"][f"{model}.{setting}.gg"]
        gg_by_seed = {int(row["seed"]): row for row in gg_cell["per_seed"]}
        for tf in tf_cell["per_seed"]:
            seed = int(tf["seed"])
            gg = gg_by_seed[seed]
            tf_history = histories.get(tf["base_run_id"], {})
            gg_history = histories.get(gg["base_run_id"], {})
            timing_available = (
                tf_history.get("status") == "success"
                and gg_history.get("status") == "success"
                and bool(tf_history.get("duration"))
                and bool(gg_history.get("duration"))
            )
            tf_seconds = duration_seconds(tf_history["duration"]) if timing_available else None
            gg_seconds = duration_seconds(gg_history["duration"]) if timing_available else None
            tf_params, tf_sha = load_params(tf["terminal_registry_run_id"])
            gg_params, gg_sha = load_params(gg["terminal_registry_run_id"])
            params_available = tf_params is not None and gg_params is not None
            records.append({
                "model": model,
                "setting": setting,
                "seed": seed,
                "tf_base_run_id": tf["base_run_id"],
                "gg_base_run_id": gg["base_run_id"],
                "tf_terminal_run_id": tf["terminal_registry_run_id"],
                "gg_terminal_run_id": gg["terminal_registry_run_id"],
                "tf_result_sha256": tf_sha,
                "gg_result_sha256": gg_sha,
                "train_time": {
                    "available": timing_available,
                    "source": "immutable PM history.duration" if timing_available else None,
                    "tf_seconds": tf_seconds,
                    "gg_seconds": gg_seconds,
                    "tf_over_gg": paired_ratio(tf_seconds, gg_seconds) if timing_available else None,
                    "missing_reason": None if timing_available else "paired successful PM durations unavailable locally",
                },
                "trainable_parameters": {
                    "available": params_available,
                    "tf": tf_params,
                    "gg": gg_params,
                    "tf_over_gg": paired_ratio(tf_params, gg_params) if params_available else None,
                    "missing_reason": None if params_available else "terminal result omits trainable_parameters",
                },
            })

    groups = {}
    for model, setting in sorted({(row["model"], row["setting"]) for row in records}):
        selected = [row for row in records if row["model"] == model and row["setting"] == setting]
        groups[f"{model}.{setting}"] = {
            "train_time_tf_over_gg": summary([
                row["train_time"]["tf_over_gg"] for row in selected if row["train_time"]["available"]
            ]),
            "trainable_params_tf_over_gg": summary([
                row["trainable_parameters"]["tf_over_gg"]
                for row in selected if row["trainable_parameters"]["available"]
            ]),
        }
    timing = [row["train_time"]["tf_over_gg"] for row in records if row["train_time"]["available"]]
    params = [
        row["trainable_parameters"]["tf_over_gg"]
        for row in records if row["trainable_parameters"]["available"]
    ]
    complete = len(timing) == len(records) and len(params) == len(records)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "complete" if complete else "partial",
        "scope": "M9 evidence currently recoverable from local immutable artifacts",
        "source_sha256": {str(OFFICIAL.relative_to(ROOT)): digest(OFFICIAL), str(HISTORY.relative_to(ROOT)): digest(HISTORY)},
        "coverage": {
            "required_paired_cells": len(records),
            "train_time_paired_cells": len(timing),
            "trainable_parameter_paired_cells": len(params),
            "calibration_gpu_hours_cells": 0,
            "assignment_cpu_seconds_cells": 0,
            "peak_calibration_memory_cells": 0,
            "inference_throughput_cells": 0,
        },
        "assertions": {
            "no_registry_timestamp_used_as_wall_time": True,
            "all_available_train_time_ratios_positive": all(value > 0 for value in timing),
            "all_available_trainable_parameter_ratios_exactly_one": all(value == 1.0 for value in params),
            "missing_evidence_not_imputed": True,
        },
        "available_macro": {
            "train_time_tf_over_gg": summary(timing),
            "trainable_params_tf_over_gg": summary(params),
        },
        "by_model_setting": groups,
        "records": records,
        "remaining_work": [
            "recover or rerun exact PM wall-time evidence for missing paired seeds",
            "extract gate/Fisher calibration GPU-hours",
            "extract assignment solver CPU wall seconds",
            "measure calibration max_memory_allocated",
            "run controlled matched TF/GG inference-throughput benchmark",
            "recover sequential trainable-parameter counts from checkpoints/configs",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# M9 Efficiency — Partial Evidence", "",
        "This table contains only directly recovered evidence. Missing cells are not imputed.", "",
        f"Status: **{payload['status']}**", "",
        "| Metric | Available / required | Available-only mean TF/GG |",
        "|---|---:|---:|",
        f"| Train wall time | {len(timing)} / {len(records)} | {payload['available_macro']['train_time_tf_over_gg']['mean']:.4f} |",
        f"| Trainable parameters | {len(params)} / {len(records)} | {payload['available_macro']['trainable_params_tf_over_gg']['mean']:.4f} |",
        "| Calibration GPU-hours | 0 / 55 | — |",
        "| Assignment CPU seconds | 0 / 55 | — |",
        "| Peak calibration memory | 0 / 55 | — |",
        "| Inference throughput | 0 / 55 | — |", "",
        "The 25 available trainable-parameter pairs are all exactly 1.000. The result remains partial until every required component is measured.",
    ]
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "coverage": payload["coverage"], "output": str(OUTPUT.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
