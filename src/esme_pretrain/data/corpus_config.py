from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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


class DocumentSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    unit: str
    rule: str
    hash_field: str
    validation_modulo: int
    validation_remainder: int
    seed: int
