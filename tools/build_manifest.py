#!/usr/bin/env python3
"""Write a deterministic SHA256 manifest for the public artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manifests" / "release_files.sha256.json"
IGNORED = {".git", ".venv", ".runs", "__pycache__", ".pytest_cache"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == OUTPUT or any(part in IGNORED for part in path.parts):
        continue
    rows.append(
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    )
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps({"files": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(OUTPUT)

