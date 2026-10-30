"""Per-launch GPU cost bounds shared by the paid launchers."""

from __future__ import annotations

import math
from dataclasses import dataclass


def reserved_cost_usd(*, timeout_seconds: float, usd_per_hour: float) -> float:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    if not math.isfinite(usd_per_hour) or usd_per_hour <= 0:
        raise ValueError("usd_per_hour must be positive and finite")
    return timeout_seconds * usd_per_hour / 3600.0


def check_spend_cap(worst_case_cost_usd: float, spend_cap_usd: float) -> None:
    if not math.isfinite(spend_cap_usd) or spend_cap_usd <= 0:
        raise ValueError("spend cap must be positive and finite")
    if not math.isfinite(worst_case_cost_usd) or worst_case_cost_usd <= 0:
        raise ValueError("worst-case cost must be positive and finite")
    if worst_case_cost_usd > spend_cap_usd:
        raise ValueError(
            f"worst-case cost ${worst_case_cost_usd:.4f} exceeds cap ${spend_cap_usd:.2f}"
        )


def project_cost(
    *, train_tokens: int, tokens_per_second: float, usd_per_hour: float
) -> float | None:
    if tokens_per_second <= 0 or not math.isfinite(tokens_per_second):
        return None
    return train_tokens / tokens_per_second * usd_per_hour / 3600.0


@dataclass(frozen=True)
class LaunchBudget:
    gpu: str
    timeout_seconds: int
    usd_per_hour: float
    spend_cap_usd: float

    def __post_init__(self) -> None:
        check_spend_cap(self.worst_case_cost_usd, self.spend_cap_usd)

    @property
    def worst_case_cost_usd(self) -> float:
        return reserved_cost_usd(
            timeout_seconds=self.timeout_seconds, usd_per_hour=self.usd_per_hour
        )
