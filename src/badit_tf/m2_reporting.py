"""Six-model M2 fidelity aggregation for the paper table."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METHOD_ORDER = ("contiguous", "random_balanced", "gg_dog", "raw_q", "tf")
MODEL_ORDER = (
    "Qwen3-4B",
    "Llama3-3B",
    "Gemma2-2B",
    "Qwen3-8B",
    "Llama3-8B",
    "Gemma2-9B",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise AssertionError(f"non-finite M2 aggregate input {label}: {number}")
    return number


def aggregate_m2_results(result_paths: list[Path], *, bootstrap_replicates: int = 1000) -> dict[str, Any]:
    if len(result_paths) != len(MODEL_ORDER):
        raise AssertionError(f"expected six M2 results, got {len(result_paths)}")
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload["model"])
        if model in loaded:
            raise AssertionError(f"duplicate M2 model: {model}")
        if payload.get("status") != "complete":
            raise AssertionError(f"M2 result is not complete: {path}")
        assertions = payload.get("metrics", {}).get("assertions", {})
        if not assertions or not all(assertions.values()):
            raise AssertionError(f"M2 terminal assertions failed: {path}")
        if set(payload["metrics"]["methods"]) != set(METHOD_ORDER):
            raise AssertionError(f"M2 method order/set mismatch: {path}")
        loaded[model] = (path, payload)
    if set(loaded) != set(MODEL_ORDER):
        raise AssertionError(f"M2 model set mismatch: {sorted(loaded)}")

    etas = {_finite(payload["eta"], f"{model}.eta") for model, (_, payload) in loaded.items()}
    split_hashes = {str(payload["split_manifest_sha256"]) for _, payload in loaded.values()}
    if len(etas) != 1 or len(split_hashes) != 1:
        raise AssertionError(f"M2 aggregate identity mismatch: etas={etas}, splits={split_hashes}")

    model_rows: dict[str, dict[str, Any]] = {}
    bootstraps: dict[str, dict[str, list[float]]] = {}
    source_artifacts = []
    for model in MODEL_ORDER:
        path, payload = loaded[model]
        bootstrap_path = Path(payload["artifacts"]["bootstrap_samples"])
        if not bootstrap_path.is_file():
            raise FileNotFoundError(bootstrap_path)
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        bootstraps[model] = bootstrap
        source_artifacts.append(
            {
                "model": model,
                "run_id": payload["run_id"],
                "result_path": str(path),
                "result_sha256": sha256_file(path),
                "bootstrap_path": str(bootstrap_path),
                "bootstrap_sha256": sha256_file(bootstrap_path),
                "git_commit": payload["git_commit"],
                "config_sha256": payload["config_sha256"],
            }
        )
        noise_p95 = _finite(payload["metrics"]["repeated_forward_noise_abs"]["p95"], f"{model}.noise_p95")
        methods = {}
        for method in METHOD_ORDER:
            metric = payload["metrics"]["methods"][method]
            values = [_finite(value, f"{model}.{method}.bootstrap") for value in bootstrap[method]]
            if len(values) != bootstrap_replicates:
                raise AssertionError(
                    f"{model}/{method} has {len(values)} bootstrap values, expected {bootstrap_replicates}"
                )
            median_gap = _finite(metric["observed_gap_median_abs"], f"{model}.{method}.median_gap")
            methods[method] = {
                **metric,
                "observed_gap_to_noise_p95_ratio": median_gap / noise_p95 if noise_p95 else float("inf"),
            }
        model_rows[model] = methods

    macro: dict[str, Any] = {}
    for method in METHOD_ORDER:
        combined_bootstrap = np.mean(
            np.asarray([bootstraps[model][method] for model in MODEL_ORDER], dtype=float), axis=0
        )
        macro[method] = {
            "predicted_regret_mean": float(np.mean([model_rows[m][method]["predicted_regret_mean"] for m in MODEL_ORDER])),
            "observed_gap_mean": float(np.mean([model_rows[m][method]["observed_gap_mean"] for m in MODEL_ORDER])),
            "observed_gap_median_abs_macro_mean": float(
                np.mean([model_rows[m][method]["observed_gap_median_abs"] for m in MODEL_ORDER])
            ),
            "spearman": float(np.mean([model_rows[m][method]["spearman"] for m in MODEL_ORDER])),
            "task_bootstrap_95ci": [
                float(np.quantile(combined_bootstrap, 0.025)),
                float(np.quantile(combined_bootstrap, 0.975)),
            ],
            "rank_accuracy": float(np.mean([model_rows[m][method]["rank_accuracy"] for m in MODEL_ORDER])),
            "top_bottom_observed_gap_separation": float(
                np.mean([model_rows[m][method]["top_bottom_observed_gap_separation"] for m in MODEL_ORDER])
            ),
            "models": len(MODEL_ORDER),
            "bootstrap_replicates": bootstrap_replicates,
        }

    per_model_tf_wins = {
        model: bool(
            model_rows[model]["tf"]["predicted_regret_mean"]
            < model_rows[model]["contiguous"]["predicted_regret_mean"]
            and model_rows[model]["tf"]["predicted_regret_mean"]
            < model_rows[model]["random_balanced"]["predicted_regret_mean"]
        )
        for model in MODEL_ORDER
    }
    assertions = {
        "six_models_present": True,
        "all_source_results_complete": True,
        "all_source_assertions_true": True,
        "common_eta": True,
        "common_split_manifest": True,
        "five_methods_present": True,
        "bootstrap_replicates_complete": True,
        "tf_tying_regret_below_contiguous_and_random_all_models": all(per_model_tf_wins.values()),
        "tf_macro_tying_regret_below_contiguous_and_random": bool(
            macro["tf"]["predicted_regret_mean"] < macro["contiguous"]["predicted_regret_mean"]
            and macro["tf"]["predicted_regret_mean"] < macro["random_balanced"]["predicted_regret_mean"]
        ),
    }
    return {
        "experiment_id": "M2-FIDELITY-SIX-MODEL-AGGREGATE",
        "run_id": "m2_fidelity_six_model_aggregate",
        "status": "complete" if all(assertions.values()) else "failed",
        "eta": next(iter(etas)),
        "split_manifest_sha256": next(iter(split_hashes)),
        "model_order": list(MODEL_ORDER),
        "method_order": list(METHOD_ORDER),
        "macro_average_definition": "arithmetic mean of the six model-level metrics; task bootstrap within each model then macro-mean per replicate",
        "metrics": {"macro": macro, "per_model": model_rows, "per_model_tf_wins": per_model_tf_wins},
        "assertions": assertions,
        "sources": source_artifacts,
    }


def write_m2_table(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = (
        "assignment",
        "predicted_regret",
        "observed_gap",
        "spearman",
        "task_bootstrap_ci_lower",
        "task_bootstrap_ci_upper",
        "rank_accuracy",
        "top_bottom_gap",
        "models",
    )
    with (output_dir / "table_fidelity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for method in METHOD_ORDER:
            row = result["metrics"]["macro"][method]
            writer.writerow(
                {
                    "assignment": method,
                    "predicted_regret": row["predicted_regret_mean"],
                    "observed_gap": row["observed_gap_mean"],
                    "spearman": row["spearman"],
                    "task_bootstrap_ci_lower": row["task_bootstrap_95ci"][0],
                    "task_bootstrap_ci_upper": row["task_bootstrap_95ci"][1],
                    "rank_accuracy": row["rank_accuracy"],
                    "top_bottom_gap": row["top_bottom_observed_gap_separation"],
                    "models": row["models"],
                }
            )
    lines = [
        "# M2 Six-Model Fidelity Results",
        "",
        f"Actual locked local scale: eta={result['eta']}. Values are six-model macro-averages.",
        "",
        "| Assignment | Pred. regret | Obs. gap | Spearman (task-bootstrap 95% CI) | Rank accuracy | Top-bottom gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = result["metrics"]["macro"][method]
        ci = row["task_bootstrap_95ci"]
        lines.append(
            f"| {method} | {row['predicted_regret_mean']:.6f} | {row['observed_gap_mean']:.6f} | "
            f"{row['spearman']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | {row['rank_accuracy']:.6f} | "
            f"{row['top_bottom_observed_gap_separation']:.6f} |"
        )
    lines.extend(["", "## Per-model raw aggregate inputs", ""])
    for model in MODEL_ORDER:
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Assignment | Pred. regret | Obs. gap | Spearman | Rank accuracy | Top-bottom gap |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for method in METHOD_ORDER:
            row = result["metrics"]["per_model"][model][method]
            lines.append(
                f"| {method} | {row['predicted_regret_mean']:.6f} | {row['observed_gap_mean']:.6f} | "
                f"{row['spearman']:.6f} | {row['rank_accuracy']:.6f} | {row['top_bottom_observed_gap_separation']:.6f} |"
            )
        lines.append("")
    (output_dir / "M2_FIDELITY_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
