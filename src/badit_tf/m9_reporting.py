"""Evidence-preserving helpers for the M9 efficiency table."""

from __future__ import annotations

import statistics


def duration_seconds(value: str) -> int:
    """Parse PM's H:MM:SS duration without using registry timestamps."""

    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError(f"invalid PM duration: {value!r}")
    hours, minutes, seconds = (int(field) for field in fields)
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid PM duration: {value!r}")
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError(f"non-positive PM duration: {value!r}")
    return total


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": None, "sample_std": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def paired_ratio(numerator: float | int, denominator: float | int) -> float:
    if numerator <= 0 or denominator <= 0:
        raise ValueError("efficiency ratios require positive values")
    return float(numerator) / float(denominator)
