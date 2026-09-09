import json
from pathlib import Path

import pytest

from badit_tf.m2_reporting import METHOD_ORDER, MODEL_ORDER, aggregate_m2_results, write_m2_table


def _result(tmp_path: Path, model: str, index: int) -> Path:
    run_dir = tmp_path / model
    run_dir.mkdir()
    bootstrap = {method: [0.5 + index * 0.01] * 1000 for method in METHOD_ORDER}
    bootstrap_path = run_dir / "m2_bootstrap_samples.json"
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    regrets = {"contiguous": 10.0, "random_balanced": 11.0, "gg_dog": 9.0, "raw_q": 8.0, "tf": 4.0}
    methods = {
        method: {
            "predicted_regret_mean": regret + index,
            "observed_gap_mean": 0.01 * regret,
            "observed_gap_median_abs": 0.001 * regret,
            "spearman": 0.5 + index * 0.01,
            "task_bootstrap_95ci": [0.4, 0.6],
            "rank_accuracy": 0.7,
            "rank_accuracy_pairs": 10000,
            "top_bottom_observed_gap_separation": 0.02 * regret,
            "gate_displacement": {"median": 0.1, "p95": 0.2, "max": 0.3},
            "units": 10,
            "decile_units": 1,
        }
        for method, regret in regrets.items()
    }
    payload = {
        "run_id": f"run_{index}",
        "model": model,
        "status": "complete",
        "git_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "eta": 0.0001,
        "metrics": {
            "methods": methods,
            "repeated_forward_noise_abs": {"median": 1e-6, "p95": 2e-6, "max": 3e-6},
            "assertions": {"complete": True},
        },
        "artifacts": {"bootstrap_samples": str(bootstrap_path)},
    }
    path = run_dir / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_six_model_macro_and_writers(tmp_path: Path) -> None:
    paths = [_result(tmp_path, model, index) for index, model in enumerate(MODEL_ORDER)]
    result = aggregate_m2_results(paths)
    assert result["status"] == "complete"
    assert result["metrics"]["macro"]["tf"]["predicted_regret_mean"] == pytest.approx(6.5)
    assert all(result["metrics"]["per_model_tf_wins"].values())
    output = tmp_path / "aggregate"
    write_m2_table(result, output)
    assert (output / "result.json").is_file()
    assert (output / "table_fidelity.csv").read_text().count("\n") == 6
    assert "Llama3-8B" in (output / "M2_FIDELITY_RESULTS.md").read_text()


def test_rejects_incomplete_source(tmp_path: Path) -> None:
    paths = [_result(tmp_path, model, index) for index, model in enumerate(MODEL_ORDER)]
    payload = json.loads(paths[0].read_text())
    payload["status"] = "failed"
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="not complete"):
        aggregate_m2_results(paths)
