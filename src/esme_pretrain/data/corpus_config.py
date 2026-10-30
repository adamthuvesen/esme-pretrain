from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class CorpusFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    min_int_score: int


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    name: str
    source: str
    revision: str
    subset: str
    split: str
    text_field: str
    id_field: str
    license: str
    language: str
    streaming: bool
    filters: CorpusFilters

    @model_validator(mode="after")
    def require_audited_recipe(self) -> Self:
        if self not in AUDITED_CORPUS_RECIPES:
            raise ValueError(
                f"corpus recipe {self.source}@{self.revision} ({self.subset}, "
                f"min_int_score={self.filters.min_int_score}) has no data audit. "
                "Write a run card with a streaming data audit (per-stage counts, "
                "contamination check) and a pre-registered ablation plan, then add the "
                "recipe to AUDITED_CORPUS_RECIPES in esme_pretrain/data/corpus_config.py"
            )
        return self


class DocumentSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    unit: str
    rule: str
    hash_field: str
    validation_modulo: int
    validation_remainder: int
    seed: int


# Each recipe maps to the run card that holds its data audit and approval.
AUDITED_CORPUS_RECIPES: dict[DatasetConfig, str] = {
    DatasetConfig.model_construct(  # pyright: ignore[reportUnhashable]
        name="fineweb-edu",
        source="HuggingFaceFW/fineweb-edu",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        subset="sample-10BT",
        split="train",
        text_field="text",
        id_field="id",
        license="odc-by",
        language="en",
        streaming=True,
        filters=CorpusFilters(min_int_score=3),
    ): "docs/run-cards/pretrain-214m-b200.md",
}
