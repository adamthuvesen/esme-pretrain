from __future__ import annotations

from typing import Any, Self

import pytest

MODAL_NAMES = ("modal", "app", "train_arm", "eval_heldout", "run_bench", "run_smoke")


class ModalRecorder:
    """Stands in for each Modal object a launcher touches, and records every call."""

    def __init__(self, calls: list[str], path: str) -> None:
        self._calls = calls
        self._path = path

    def __getattr__(self, name: str) -> ModalRecorder:
        return ModalRecorder(self._calls, f"{self._path}.{name}")

    def __call__(self, *_args: Any, **_kwargs: Any) -> ModalRecorder:
        self._calls.append(self._path)
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


@pytest.fixture
def modal_calls(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace a launcher module's Modal objects with recorders; return the call list."""
    calls: list[str] = []

    def install(module: object) -> list[str]:
        for name in MODAL_NAMES:
            monkeypatch.setattr(module, name, ModalRecorder(calls, name), raising=False)
        return calls

    return install
