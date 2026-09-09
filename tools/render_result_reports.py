#!/usr/bin/env python3
"""Render structured Excel and Markdown reports from the sole registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from badit_tf.reporting import render_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = render_reports(args.registry, args.xlsx, args.markdown)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
