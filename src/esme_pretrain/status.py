from __future__ import annotations

# Keep in sync with docs/status.md.
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStage:
    order: int
    name: str
    milestone: str
    status: str


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(1, "raw text", "local corpus fixtures and FineWeb-Edu streaming", "available"),
    PipelineStage(2, "data report", "dataset accounting and split report", "available"),
    PipelineStage(3, "tokenizer", "214M B200 byte-level BPE contract", "accepted"),
    PipelineStage(4, "packed tokens", "fixed-window token shards", "available"),
    PipelineStage(5, "transformer", "214M conventional GQA pretrain model", "accepted"),
    PipelineStage(6, "training loop", "B200 Modal pretrain loop", "accepted 10B run"),
    PipelineStage(7, "eval", "fixed checkpoint eval + bpb", "accepted"),
    PipelineStage(8, "scaling curves", "optional scaling mini-series", "available later"),
    PipelineStage(9, "checkpoint export", "llm-infer bundle export", "accepted"),
)

SIBLING_REPOS: dict[str, str] = {
    "esme-posttrain": "SFT + simple-task RL on the from-scratch checkpoint",
    "llm-rlvr": "adapt a model with SFT and execution-verified RLVR",
    "grpo-decomp": "measure where RLVR gains come from",
    "llm-infer": "serve and benchmark exported checkpoints",
}


@dataclass(frozen=True)
class ProjectStatus:
    state: str
    summary: str
    pipeline: tuple[PipelineStage, ...]
    next_milestone: str
    spend_policy: str
    run_card_path: str


def current_status() -> ProjectStatus:
    return ProjectStatus(
        state="214M B200 pretrain accepted",
        summary=(
            "The 214M conventional GQA model completed the accepted 10B FineWeb-Edu B200 "
            "pretrain. Fixed checkpoint evaluation, base acceptance reporting, bits-per-byte "
            "reporting, and Esme-214M-Base export for llm-infer are complete."
        ),
        pipeline=PIPELINE_STAGES,
        next_milestone=(
            "Posttraining approval and work in esme-posttrain, not another pretrain launch."
        ),
        spend_policy=(
            "No FineWeb-Edu, ClimbMix, Modal, GPU, or paid compute without the relevant run "
            "card and explicit approval."
        ),
        run_card_path="docs/run-cards/pretrain-214m-b200.md",
    )
