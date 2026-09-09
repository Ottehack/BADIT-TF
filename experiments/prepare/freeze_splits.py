#!/usr/bin/env python3
"""Freeze BADIT-TF calibration and tuning manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from badit_tf.splits import SplitCounts, build_split_manifest, write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/SuperNI"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/configs/splits/default_seed1.json"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--assignment", type=int, default=16)
    parser.add_argument("--fisher", type=int, default=16)
    parser.add_argument("--damping-validation", type=int, default=8)
    parser.add_argument("--fidelity", type=int, default=16)
    args = parser.parse_args()
    counts = SplitCounts(
        assignment=args.assignment,
        fisher=args.fisher,
        damping_validation=args.damping_validation,
        fidelity=args.fidelity,
    )
    manifest = build_split_manifest(args.data_root, seed=args.seed, counts=counts)
    write_split_manifest(manifest, args.output)
    print(args.output)
    print(manifest["manifest_sha256"])
    print({role: len(rows) for role, rows in manifest["roles"].items()})


if __name__ == "__main__":
    main()

