#!/usr/bin/env python3
"""Turn measured throughput into cost-per-billion-token projections.

Reads the JSON written by scripts/modal_throughput_probe.py and projects, for each
model size, the cost-per-billion-tokens and the full-pretrain cost at a
compute-optimal (20 tok/param) and an over-trained (100 tok/param) budget. Keeps
the numbers in docs/internal/scaleup-probe.md reproducible.

    uv run python scripts/probe_projections.py \
        --results runs/stage0-throughput-probe/probe-results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Modal on-demand A100-40GB list price, 2026-06 (modal.com/pricing): $0.000583/s.
A100_40GB_USD_PER_SEC = 0.000583
A100_80GB_USD_PER_SEC = 0.000694


def _projection_for(result: dict, usd_per_sec: float, realization: float) -> dict:
    tps_ceiling = result["tokens_per_second"]
    tps_real = tps_ceiling * realization
    total_params = result["parameter_total"]

    def pretrain(tokens_per_param: int) -> dict:
        tokens = tokens_per_param * total_params
        seconds = tokens / tps_real
        return {
            "tokens_per_param": tokens_per_param,
            "tokens": tokens,
            "tokens_billion": round(tokens / 1e9, 2),
            "wall_clock_hours": round(seconds / 3600, 2),
            "cost_usd": round(seconds * usd_per_sec, 2),
        }

    cost_per_b_ceiling = (1e9 / tps_ceiling) * usd_per_sec
    cost_per_b_real = (1e9 / tps_real) * usd_per_sec
    return {
        "model": result["model_name"],
        "parameter_total": total_params,
        "tokens_per_second_ceiling": round(tps_ceiling),
        "tokens_per_second_realistic": round(tps_real),
        "mfu": result["mfu"],
        "step_time_ms": round(result["step_time_ms"], 1),
        "cost_per_billion_tokens_ceiling_usd": round(cost_per_b_ceiling, 2),
        "cost_per_billion_tokens_realistic_usd": round(cost_per_b_real, 2),
        "compute_optimal_20x": pretrain(20),
        "overtrained_100x": pretrain(100),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--usd-per-sec",
        type=float,
        default=A100_40GB_USD_PER_SEC,
        help="GPU price per second (default Modal A100-40GB).",
    )
    parser.add_argument(
        "--realization",
        type=float,
        default=0.9,
        help="Fraction of compute-ceiling tokens/sec a real run sustains.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    projections = [
        _projection_for(row, args.usd_per_sec, args.realization)
        for row in payload["results"]
        if "compiled" not in row["model_name"]
    ]

    if args.json:
        print(json.dumps({"projections": projections}, indent=2))
        return 0

    print(f"GPU price: ${args.usd_per_sec}/s  realization factor: {args.realization}\n")
    print(
        f"{'model':>8} {'params':>12} {'tok/s(real)':>12} {'MFU':>6} "
        f"{'$/B tok':>9} {'20x: B tok':>11} {'20x: $':>8} {'100x: $':>9}"
    )
    for p in projections:
        mfu = f"{p['mfu'] * 100:.1f}%" if p["mfu"] is not None else "n/a"
        print(
            f"{p['model']:>8} {p['parameter_total']:>12,} "
            f"{p['tokens_per_second_realistic']:>12,} {mfu:>6} "
            f"{p['cost_per_billion_tokens_realistic_usd']:>9.2f} "
            f"{p['compute_optimal_20x']['tokens_billion']:>11.2f} "
            f"{p['compute_optimal_20x']['cost_usd']:>8.2f} "
            f"{p['overtrained_100x']['cost_usd']:>9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
