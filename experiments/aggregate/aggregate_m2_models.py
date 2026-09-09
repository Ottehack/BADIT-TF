#!/usr/bin/env python3
"""Aggregate the six frozen M2 model results into tab:fidelity."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from badit_tf.m2_reporting import aggregate_m2_results, write_m2_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = aggregate_m2_results(
        [Path(item["result_path"]) for item in config["inputs"]],
        bootstrap_replicates=int(config.get("bootstrap_replicates", 1000)),
    )
    write_m2_table(result, Path(config["output_dir"]))
    if result["status"] != "complete":
        raise SystemExit(1)
    print(Path(config["output_dir"]) / "result.json")


if __name__ == "__main__":
    main()
