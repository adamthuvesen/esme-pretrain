"""Resolve study input files relative to the mounted study volume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esme_pretrain.study.plan import StudyData


def relative_volume_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path == PurePosixPath(".")
        or "\\" in value
    ):
        raise ValueError(
            f"got {value!r}, expected a relative file path inside the study volume. "
            "Use a POSIX path without '..'."
        )
    return path.as_posix()


@dataclass(frozen=True)
class StudyDataPaths:
    root: Path
    tokenizer: Path
    train_manifest: Path
    validation_manifest: Path


def resolve_study_data(data: StudyData, root: Path) -> StudyDataPaths:
    root = root.resolve()
    paths = {}
    for field in ("tokenizer", "train_manifest", "validation_manifest"):
        relative = relative_volume_path(getattr(data, field))
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"data.{field}: {relative!r} resolves outside {root}. "
                "Select a file inside the study volume."
            )
        paths[field] = resolved
    return StudyDataPaths(
        root=root,
        tokenizer=paths["tokenizer"],
        train_manifest=paths["train_manifest"],
        validation_manifest=paths["validation_manifest"],
    )
