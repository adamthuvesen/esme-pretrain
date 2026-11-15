"""The study data config and its input files resolved inside the mounted study volume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, field_validator


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


class StudySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class StudyData(StudySettings):
    volume: str
    train_manifest: str
    validation_manifest: str
    tokenizer: str
    shuffle_seed: int

    @field_validator("train_manifest", "validation_manifest", "tokenizer")
    @classmethod
    def safe_volume_path(cls, value: str) -> str:
        return relative_volume_path(value)


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
