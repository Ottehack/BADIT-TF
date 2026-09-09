#!/usr/bin/env python3
"""Fail-closed H1 audit and selection for local ModelScope fallback executions.

The v2 H1 grid freezes 34 rows for each owner-authorized ModelScope model.
This reader never creates a lock until every row has a complete, registry-backed,
validation-only result with the frozen config hash and passing assertions.  A
failed local attempt remains in the audit; a registered ``rN`` retry may supply
the successful result for exactly the same immutable grid configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl


MODELS = {"Llama3-3B": "llama3_3b", "Gemma2-2B": "gemma2_2b"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def local_run_id(grid_run_id: str) -> str:
    return grid_run_id.replace("h1-", "h1_") + "_local_a100"


def retry_run_id(base_run_id: str, retry_tag: str) -> str:
    return base_run_id.replace("_local_a100", f"_{retry_tag}_local_a100")


def resolve_result_path(root: Path, raw_result_path: str) -> Path:
    """Normalize legacy directory-valued registry paths to ``result.json``.

    Formal local H1 r1 rows predating the remote collector recorded their raw
    result directory, while later rows record the JSON file itself.  Both are
    immutable artifacts; accepting the directory spelling only when its
    ``result.json`` exists avoids treating a checkpoint-only directory as a
    completed evaluator output.
    """
    path = root / raw_result_path
    return path / "result.json" if path.is_dir() else path


def registry_rows(registry: Path) -> dict[str, dict[str, Any]]:
    workbook = openpyxl.load_workbook(registry, data_only=True, read_only=True)
    rows: dict[str, dict[str, Any]] = {}
    for sheet in workbook.worksheets:
        values = sheet.iter_rows(values_only=True)
        headers = next(values, None)
        if not headers or "run_id" not in headers:
            continue
        names = [str(value) if value is not None else "" for value in headers]
        for row in values:
            record = dict(zip(names, row, strict=False))
            run_id = record.get("run_id")
            if not run_id:
                continue
            require(str(run_id) not in rows, f"duplicate registry run ID: {run_id}")
            rows[str(run_id)] = record
    return rows


def remote_run_id(grid_run_id: str) -> str:
    """Return the pre-registered source ID used by the remote L20Z wave."""
    return grid_run_id.replace("h1-", "h1_") + "_remote_l20z"


def candidate_ids(grid_run_id: str, registry: dict[str, dict[str, Any]]) -> list[str]:
    """Find one completed retry or source run for a frozen grid cell.

    A grid cell may be fulfilled by exactly one execution.  Accepting both a
    completed local and remote copy would silently turn the fixed grid into a
    post-hoc replicate selection, so the caller fails closed on that case.
    """
    base_run_id = local_run_id(grid_run_id)
    local_prefix = base_run_id[:-len("_local_a100")]
    local_pattern = re.compile(rf"^{re.escape(local_prefix)}(?:_r[1-9][0-9]*)?_local_a100$")
    source_remote_id = remote_run_id(grid_run_id)
    remote_prefix = source_remote_id[:-len("_remote_l20z")]
    remote_pattern = re.compile(rf"^{re.escape(remote_prefix)}(?:_r[1-9][0-9]*)?_remote_l20z$")
    return sorted(
        run_id
        for run_id, registry_row in registry.items()
        if registry_row.get("status") == "complete"
        and (bool(local_pattern.fullmatch(run_id)) or bool(remote_pattern.fullmatch(run_id)))
    )


def attempt_history(grid_run_id: str, registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base_run_id = local_run_id(grid_run_id)
    prefix = base_run_id[:-len("_local_a100")]
    pattern = re.compile(rf"^{re.escape(prefix)}(?:_r[1-9][0-9]*)?_local_a100$")
    source_remote_id = remote_run_id(grid_run_id)
    remote_prefix = source_remote_id[:-len("_remote_l20z")]
    remote_pattern = re.compile(rf"^{re.escape(remote_prefix)}(?:_r[1-9][0-9]*)?_remote_l20z$")
    history = []
    for run_id, row in sorted(registry.items()):
        if pattern.fullmatch(run_id) or remote_pattern.fullmatch(run_id):
            history.append(
                {
                    "run_id": run_id,
                    "status": row.get("status"),
                    "failure_reason": row.get("failure_reason") or None,
                    "raw_result_path": row.get("raw_result_path"),
                }
            )
    return history


def choose(rows: list[dict[str, Any]], setting: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = sorted(rows, key=lambda item: (-item["primary_score"], item["trial_id"]))
    for rank, row in enumerate(ranked, start=1):
        row["primary_rank"] = rank
    if setting == "mixed":
        return ranked[0], ranked
    best_score = ranked[0]["primary_score"]
    tie_band = [row for row in ranked if best_score - row["primary_score"] <= 0.1]
    return min(tie_band, key=lambda item: (item["forget_rate"], -item["backward"], item["trial_id"])), ranked


def audit_row(root: Path, row: dict[str, Any], registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base_run_id = local_run_id(str(row["run_id"]))
    candidates = candidate_ids(str(row["run_id"]), registry)
    require(len(candidates) == 1, f"expected one complete candidate for {base_run_id}, found {candidates}")
    run_id = candidates[0]
    registry_row = registry[run_id]
    result_path = resolve_result_path(root, str(registry_row["raw_result_path"]))
    require(result_path.is_file(), f"missing result {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))

    require(result.get("run_id") == run_id, f"result run ID mismatch: {run_id}")
    require(result.get("status") == "complete", f"non-complete result: {run_id}")
    require(result.get("config_path") == row["config_path"], f"config path mismatch: {run_id}")
    require(result.get("config_sha256") == row["config_sha256"], f"config SHA mismatch: {run_id}")
    config_path = root / str(row["config_path"])
    require(config_path.is_file(), f"missing config {config_path}")
    require(sha256_file(config_path) == row["config_sha256"], f"config file SHA mismatch: {run_id}")

    protocol = result.get("protocol", {})
    require(protocol.get("evaluation_role") == "tune_validation", f"non-validation evaluator: {run_id}")
    if row["setting"] == "mixed":
        require(protocol.get("official_test_loaded") is False, f"official test loaded: {run_id}")
        require(protocol.get("final_test_used_for_selection") is False, f"test selection enabled: {run_id}")
        test_closure_evidence = "official_test_loaded=false; final_test_used_for_selection=false"
    else:
        # run_p3_sequential is an older evaluator schema: it records the
        # authoritative no-test-selection field rather than the mixed runner's
        # official_test_loaded/final_test_used_for_selection pair.
        require(
            protocol.get("official_test_used_for_selection") is False,
            f"test selection enabled or unrecorded: {run_id}",
        )
        test_closure_evidence = "official_test_used_for_selection=false"
    assertions = result.get("metrics", {}).get("assertions", {})
    require(bool(assertions), f"missing assertions: {run_id}")
    failed_assertions = sorted(key for key, value in assertions.items() if value is not True)
    require(not failed_assertions, f"failed assertions in {run_id}: {failed_assertions}")

    audited: dict[str, Any] = {
        "model": row["model"],
        "setting": row["setting"],
        "trial_id": row["trial_id"],
        "changed_axis": row["changed_axis"],
        "run_id": run_id,
        "base_run_id": base_run_id,
        "attempt_history": attempt_history(str(row["run_id"]), registry),
        "config_path": row["config_path"],
        "config_sha256": row["config_sha256"],
        "result_path": str(result_path.relative_to(root)),
        "result_sha256": sha256_file(result_path),
        "code_commit": result.get("code_commit"),
        "evaluation_role": "tune_validation",
        "official_test_loaded": False,
        "official_test_closure_evidence": test_closure_evidence,
        "test_used_for_selection": False,
        "all_assertions_passed": True,
    }
    for field in ("learning_rate", "router_lr_multiplier", "warmup_ratio", "lora_dropout", "weight_decay"):
        audited[field] = row[field]

    metrics = result["metrics"]
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
            require(isinstance(continual.get(key), (int, float)), f"missing {key}: {run_id}")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--registry", type=Path, default=Path("experiments/experiment_registry.xlsx"))
    parser.add_argument("--grid", type=Path, default=Path("experiments/configs/h1_one_factor_grid_v2.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    grid_path = root / args.grid
    registry = registry_rows(root / args.registry)
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    rows = [row for row in grid["trial_rows"] if row["model"] == args.model]
    require(len(rows) == 34, f"expected 34 grid rows for {args.model}, found {len(rows)}")
    audited = [audit_row(root, row, registry) for row in rows]

    settings: dict[str, Any] = {}
    for setting in ("mixed", "sequential"):
        setting_rows = [row for row in audited if row["setting"] == setting]
        require(len(setting_rows) == 17, f"expected 17 {setting} rows")
        selected, ranked = choose(setting_rows, setting)
        settings[setting] = {"selected": selected, "ranked_rows": ranked}

    timestamp = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": 1,
        "experiment_id": "H1-ONE-FACTOR-MODELSCOPE",
        "model": args.model,
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
            "sequential": "maximize continual score; within 0.1 point minimize Forget Rate, then maximize Backward, then trial_id ascending",
        },
    }
    payload = {**common, "settings": settings, "audited_rows": audited}
    lock = {
        **common,
        "selected": {
            setting: {
                key: value
                for key, value in section["selected"].items()
                if key in {"trial_id", "run_id", "primary_metric", "primary_score", "forget_rate", "backward", "config_path", "config_sha256", "result_path", "result_sha256", "code_commit"}
            }
            for setting, section in settings.items()
        },
    }
    slug = MODELS[args.model]
    output_dir = root / (args.output_dir or Path(f"experiments/raw_results/h1_{slug}_one_factor_aggregate"))
    require(not output_dir.exists(), f"refusing to overwrite aggregate directory {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "selection_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        display_dir = str(output_dir.relative_to(root))
    except ValueError:
        # ``--output-dir`` is also useful for a disposable independent audit
        # outside the checkout; the output itself is already complete here.
        display_dir = str(output_dir)
    print(json.dumps({"output_dir": display_dir, "selected": lock["selected"]}, indent=2))


if __name__ == "__main__":
    main()
