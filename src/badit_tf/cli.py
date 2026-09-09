"""Portable command-line entry point for the BADIT-TF experiment artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs" / "recipes.yaml"
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
FORBIDDEN = re.compile(
    r"/chatgpt" + r"_nas|/primus" + r"_datasets|/root/\.codex|"
    r"oss:" + r"//|PM_" + r"HOST|"
    r"(?:^|[^0-9])33\.[0-9]+\.[0-9]+\.[0-9]+|remote_" + r"pm/"
)
SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?:access[_-]?key|secret[_-]?key|api[_-]?key|token|password)\s*[:=]\s*['\"][^$<{][^'\"]{7,}['\"]",
    re.IGNORECASE,
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in os.environ:
            raise ValueError(f"environment variable {key} is required")
        return os.environ[key]

    return VARIABLE.sub(replace, value)


def resolve_configs(paths: list[Path]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for path in paths:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"config must be a mapping: {path}")
        payload = _deep_merge(payload, loaded)
    return _expand(payload)


def write_resolved(config: dict[str, Any], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _catalog() -> dict[str, Any]:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["recipes"]


def command_list(_: argparse.Namespace) -> int:
    for name, recipe in _catalog().items():
        phases = ", ".join(recipe["phases"])
        print(f"{name:14} {phases:36} {recipe['description']}")
    return 0


def command_materialize(args: argparse.Namespace) -> int:
    _load_dotenv(ROOT / ".env")
    config = resolve_configs(args.config)
    output = args.output or ROOT / ".runs" / "resolved.yaml"
    write_resolved(config, output)
    print(output)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    _load_dotenv(ROOT / ".env")
    checks: dict[str, Any] = {
        "python": sys.version.split()[0],
        "repository_root": str(ROOT),
        "torchrun": shutil.which("torchrun"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    failures: list[str] = []
    try:
        import torch

        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        checks["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - environment dependent
        checks["torch_error"] = repr(exc)
        failures.append("PyTorch import failed")
    if args.config:
        try:
            config = resolve_configs(args.config)
            checks["config_keys"] = len(config)
            for key in ("model_path", "data_root"):
                if key in config:
                    exists = Path(config[key]).expanduser().exists()
                    checks[f"{key}_exists"] = exists
                    if not exists:
                        failures.append(f"{key} does not exist: {config[key]}")
        except Exception as exc:
            checks["config_error"] = repr(exc)
            failures.append("config resolution failed")
    checks["status"] = "PASS" if not failures else "FAIL"
    checks["failures"] = failures
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if not failures else 2


def command_run(args: argparse.Namespace) -> int:
    _load_dotenv(ROOT / ".env")
    recipes = _catalog()
    if args.recipe not in recipes:
        raise SystemExit(f"unknown recipe: {args.recipe}")
    phases = recipes[args.recipe]["phases"]
    if args.phase not in phases:
        raise SystemExit(f"unknown phase {args.phase!r}; choose from {', '.join(phases)}")
    phase = phases[args.phase]
    script = ROOT / phase["script"]
    if not script.is_file():
        raise FileNotFoundError(script)
    resolved = resolve_configs(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_path = write_resolved(
        resolved, ROOT / ".runs" / f"{args.recipe}_{args.phase}_{stamp}.yaml"
    )
    launcher = phase["launcher"]
    if launcher == "torchrun":
        world = args.world_size or int(resolved.get("world_size", os.environ.get("NPROC_PER_NODE", 1)))
        command = ["torchrun", "--standalone", f"--nproc-per-node={world}", str(script)]
    else:
        command = [sys.executable, str(script)]
    if not args.no_config:
        command.extend(["--config", str(config_path)])
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    command.extend(extra)
    print(json.dumps({"command": command, "resolved_config": str(config_path)}, indent=2))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def command_audit(_: argparse.Namespace) -> int:
    findings: list[dict[str, Any]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", ".venv", ".runs"} for part in path.parts):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".xlsx", ".pt", ".npz"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN.search(line):
                findings.append({"kind": "internal_path_or_host", "file": str(path.relative_to(ROOT)), "line": number})
            if SECRET.search(line):
                findings.append({"kind": "possible_secret", "file": str(path.relative_to(ROOT)), "line": number})
    print(json.dumps({"status": "PASS" if not findings else "FAIL", "findings": findings}, indent=2))
    return 0 if not findings else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="badit-tf")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=command_list)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--config", action="append", type=Path, default=[])
    doctor.set_defaults(func=command_doctor)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--config", action="append", type=Path, required=True)
    materialize.add_argument("--output", type=Path)
    materialize.set_defaults(func=command_materialize)
    run = sub.add_parser("run")
    run.add_argument("recipe")
    run.add_argument("--phase", required=True)
    run.add_argument("--config", action="append", type=Path, required=True)
    run.add_argument("--world-size", type=int)
    run.add_argument("--no-config", action="store_true", help="phase has no --config option")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=command_run)
    sub.add_parser("audit").set_defaults(func=command_audit)
    return parser


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if args.command == "run":
        args.extra = unknown
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
