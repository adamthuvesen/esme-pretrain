"""Shared JSON config validation helpers."""

from __future__ import annotations

from typing import Any


def require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{label} has unsupported keys: {', '.join(extra)}")


def expect_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value
