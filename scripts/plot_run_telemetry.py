#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly>=6.1",
#   "kaleido>=1.0",
# ]
# ///
"""Render README training-telemetry figures from pretrain run artifacts.

Reads `metrics.jsonl`, `throughput.csv`, and `run-summary.json` from a run
directory (downloaded read-only from the run's Modal volume) and exports two
static SVG cards into `assets/`:

- fig-pretrain-loss-vs-tokens.svg: train + validation loss vs training tokens.
- fig-pretrain-throughput-mfu.svg: tokens/sec stability with an MFU axis.

    uv run scripts/plot_run_telemetry.py \
        --run-dir runs/pretrain-214m-b200/pretrain_214m_b200 --output-dir assets --json

Fetch the inputs from the Modal volume (the volume kept its pre-rename name):

    modal volume get llm-pretrain-214m-b200 pretrain_214m_b200/metrics.jsonl <run-dir>/
    modal volume get llm-pretrain-214m-b200 pretrain_214m_b200/throughput.csv <run-dir>/
    modal volume get llm-pretrain-214m-b200 pretrain_214m_b200/run-summary.json <run-dir>/
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go

# Style contract: match grpo-decomp/results/fig-gsm8k-decomposition.svg.
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
TITLE_COLOR = "#1f2937"
SUBTITLE_COLOR = "#6b7280"
AXIS_COLOR = "#cfd5df"
GRID_COLOR = "#eef1f6"
BORDER_COLOR = "#d9dde7"
TICK_COLOR = "#6b7280"
BLUE = "#636efa"
GREEN = "#00cc96"
RED = "#ef553b"
CARD_WIDTH = 920
CARD_HEIGHT = 560

# bf16 dense peak for the run's GPU; mirrors DEVICE_PEAK_TFLOPS["B200"] in
# src/esme_pretrain/training/device_profile.py (this script runs in an
# isolated uv environment, so the value is repeated here).
B200_PEAK_TFLOPS = 2250.0

THROUGHPUT_MEDIAN_WINDOW = 25  # 25 samples x 10-step cadence = 250 steps.


@dataclass(frozen=True)
class RunTelemetry:
    batch_tokens: int
    train_steps: list[int]
    train_loss: list[float]
    val_steps: list[int]
    val_loss: list[float]
    throughput_steps: list[int]
    tokens_per_second: list[float]
    flops_per_token: float

    def tokens_at(self, step: int) -> float:
        return (step + 1) * self.batch_tokens

    def mfu_percent(self, tokens_per_second: float) -> float:
        achieved_tflops = tokens_per_second * self.flops_per_token / 1e12
        return achieved_tflops / B200_PEAK_TFLOPS * 100.0


def load_telemetry(run_dir: Path) -> RunTelemetry:
    metrics_path = run_dir / "metrics.jsonl"
    throughput_path = run_dir / "throughput.csv"
    summary_path = run_dir / "run-summary.json"
    for path in (metrics_path, throughput_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing run artifact: {path}")

    train_steps: list[int] = []
    train_loss: list[float] = []
    val_steps: list[int] = []
    val_loss: list[float] = []
    batch_tokens_values: set[int] = set()
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
        record = json.loads(line)
        step = record["step"]
        if "val_loss" in record:
            val_steps.append(step)
            val_loss.append(record["val_loss"])
        elif "loss" in record:
            train_steps.append(step)
            train_loss.append(record["loss"])
            batch_tokens_values.add(record["tokens"])
        else:
            raise ValueError(f"{metrics_path}:{line_number}: record has neither loss nor val_loss")
    if not train_steps or not val_steps:
        raise ValueError(f"{metrics_path}: expected both train and validation records")
    if len(batch_tokens_values) != 1:
        raise ValueError(
            f"{metrics_path}: expected constant tokens/step, got {batch_tokens_values}"
        )

    throughput_steps: list[int] = []
    tokens_per_second: list[float] = []
    with throughput_path.open() as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            if step == 0:
                continue  # First step is dominated by torch.compile warmup.
            throughput_steps.append(step)
            tokens_per_second.append(float(row["tokens_per_second"]))
    if not throughput_steps:
        raise ValueError(f"{throughput_path}: no post-warmup throughput samples")

    summary = json.loads(summary_path.read_text())
    flops_per_token = summary["flops_per_token"]
    if not flops_per_token or flops_per_token <= 0:
        raise ValueError(f"{summary_path}: flops_per_token must be positive")

    return RunTelemetry(
        batch_tokens=batch_tokens_values.pop(),
        train_steps=train_steps,
        train_loss=train_loss,
        val_steps=val_steps,
        val_loss=val_loss,
        throughput_steps=throughput_steps,
        tokens_per_second=tokens_per_second,
        flops_per_token=flops_per_token,
    )


def rolling_median(values: list[float], window: int) -> list[float]:
    half = window // 2
    return [
        statistics.median(values[max(0, i - half) : min(len(values), i + half + 1)])
        for i in range(len(values))
    ]


def rounded_border_path(radius_px: float) -> str:
    """Rounded-rect border in paper coordinates (plotly paths have no arc command)."""
    rx = radius_px / CARD_WIDTH
    ry = radius_px / CARD_HEIGHT
    x0, x1 = 0.5 / CARD_WIDTH, 1 - 0.5 / CARD_WIDTH
    y0, y1 = 0.5 / CARD_HEIGHT, 1 - 0.5 / CARD_HEIGHT
    return (
        f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
        f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
        f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
        f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"
    )


def card_layout(title: str, subtitle: str, conclusion: str) -> go.Layout:
    return go.Layout(
        width=CARD_WIDTH,
        height=CARD_HEIGHT,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": FONT_FAMILY, "size": 12, "color": TICK_COLOR},
        margin={"l": 84, "r": 84, "t": 118, "b": 96},
        showlegend=False,
        shapes=[
            {
                "type": "path",
                "path": rounded_border_path(radius_px=8),
                "xref": "paper",
                "yref": "paper",
                "line": {"color": BORDER_COLOR, "width": 1},
                "layer": "above",
            }
        ],
        annotations=[
            {
                "text": f"<b>{title}</b>",
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": 1.24,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 22, "color": TITLE_COLOR},
            },
            {
                "text": subtitle,
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": 1.135,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 14, "color": SUBTITLE_COLOR},
            },
            {
                "text": conclusion,
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": -0.185,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 12, "color": SUBTITLE_COLOR},
            },
        ],
    )


def styled_axis(**overrides: object) -> dict[str, object]:
    axis: dict[str, object] = {
        "showgrid": True,
        "gridcolor": GRID_COLOR,
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": AXIS_COLOR,
        "linewidth": 1,
        "ticks": "outside",
        "tickcolor": AXIS_COLOR,
        "tickfont": {"family": FONT_FAMILY, "size": 12, "color": TICK_COLOR},
        "title": {"font": {"family": FONT_FAMILY, "size": 13, "color": "#374151"}},
    }
    axis.update(overrides)
    return axis


def value_annotation(text: str, x: float, y: float, ax: int, ay: int) -> dict[str, object]:
    return {
        "text": f"<b>{text}</b>",
        "x": x,
        "y": y,
        "ax": ax,
        "ay": ay,
        "showarrow": True,
        "arrowcolor": AXIS_COLOR,
        "arrowwidth": 1,
        "arrowhead": 0,
        "font": {"family": FONT_FAMILY, "size": 13, "color": TITLE_COLOR},
    }


def token_axis_ticks(max_tokens: float) -> tuple[list[float], list[str]]:
    tickvals = [i * 2e9 for i in range(int(max_tokens // 2e9) + 1)]
    ticktext = [f"{val / 1e9:.0f}B" if val else "0" for val in tickvals]
    ticktext[-1] += " tokens"
    return tickvals, ticktext


def build_loss_figure(telemetry: RunTelemetry) -> go.Figure:
    train_tokens = [telemetry.tokens_at(step) for step in telemetry.train_steps]
    val_tokens = [telemetry.tokens_at(step) for step in telemetry.val_steps]
    final_val = telemetry.val_loss[-1]
    total_tokens = telemetry.tokens_at(telemetry.train_steps[-1])

    figure = go.Figure(
        layout=card_layout(
            title="Esme-214M-Base: loss over the 10B-token pretrain",
            subtitle=(
                "Train and validation cross-entropy vs training tokens - FineWeb-Edu"
                " sample-10BT, 26,015 steps, one B200"
            ),
            conclusion=(
                "Conclusion: validation loss tracks training loss down to"
                f" {final_val:.2f} with no divergence through the full"
                f" {total_tokens / 1e9:.2f}B-token budget."
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=train_tokens,
            y=telemetry.train_loss,
            mode="lines",
            name="train loss",
            line={"color": BLUE, "width": 1.4},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=val_tokens,
            y=telemetry.val_loss,
            mode="lines+markers",
            name="validation loss",
            line={"color": RED, "width": 2},
            marker={"size": 5, "color": RED},
        )
    )
    tickvals, ticktext = token_axis_ticks(total_tokens)
    figure.update_layout(
        xaxis=styled_axis(range=[0, total_tokens * 1.04], tickvals=tickvals, ticktext=ticktext),
        yaxis=styled_axis(title={"text": "loss"}, range=[2, 11]),
    )
    figure.add_annotation(
        value_annotation(f"val {final_val:.2f}", x=val_tokens[-1], y=final_val, ax=-6, ay=-42)
    )
    figure.add_annotation(
        {
            "text": "train (every 10 steps)",
            "x": total_tokens * 0.42,
            "y": 4.6,
            "showarrow": False,
            "xanchor": "left",
            "font": {"family": FONT_FAMILY, "size": 13, "color": BLUE},
        }
    )
    figure.add_annotation(
        {
            "text": "validation (every 500 steps)",
            "x": total_tokens * 0.42,
            "y": 4.0,
            "showarrow": False,
            "xanchor": "left",
            "font": {"family": FONT_FAMILY, "size": 13, "color": RED},
        }
    )
    return figure


def build_throughput_figure(telemetry: RunTelemetry) -> go.Figure:
    tokens = [telemetry.tokens_at(step) for step in telemetry.throughput_steps]
    smoothed = rolling_median(telemetry.tokens_per_second, THROUGHPUT_MEDIAN_WINDOW)
    steady = statistics.median(telemetry.tokens_per_second)
    steady_mfu = telemetry.mfu_percent(steady)
    tps_ceiling = 260_000.0

    figure = go.Figure(
        layout=card_layout(
            title="Esme-214M-Base: throughput held steady end to end",
            subtitle=(
                "Tokens/sec per 10-step window with rolling median - MFU vs the"
                f" {B200_PEAK_TFLOPS:,.0f} TFLOP/s B200 bf16 peak - compile step excluded"
            ),
            conclusion=(
                f"Conclusion: throughput stayed near {steady / 1e3:.0f}K tokens/s"
                f" ({steady_mfu:.1f}% MFU) for the whole run; dips are brief and"
                " recover immediately, with no sustained degradation."
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=tokens,
            y=telemetry.tokens_per_second,
            mode="lines",
            name="tokens/sec",
            line={"color": GREEN, "width": 1},
            opacity=0.35,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=tokens,
            y=smoothed,
            mode="lines",
            name="rolling median (250 steps)",
            line={"color": GREEN, "width": 2.4},
        )
    )
    tickvals, ticktext = token_axis_ticks(tokens[-1])
    tps_ticks = [0, 50_000, 100_000, 150_000, 200_000, 250_000]
    mfu_ticks = [0, 4, 8, 12, 16]
    figure.update_layout(
        xaxis=styled_axis(range=[0, tokens[-1] * 1.04], tickvals=tickvals, ticktext=ticktext),
        yaxis=styled_axis(
            title={"text": "tokens/sec"},
            range=[0, tps_ceiling],
            tickvals=tps_ticks,
            ticktext=[f"{val // 1000}K" if val else "0" for val in tps_ticks],
        ),
        yaxis2=styled_axis(
            title={"text": "MFU"},
            overlaying="y",
            side="right",
            range=[0, telemetry.mfu_percent(tps_ceiling)],
            tickvals=mfu_ticks,
            ticktext=[f"{val}%" for val in mfu_ticks],
            showgrid=False,
        ),
    )
    # Invisible anchor trace so the right-hand MFU axis is rendered.
    figure.add_trace(
        go.Scatter(
            x=[tokens[0]],
            y=[telemetry.mfu_percent(telemetry.tokens_per_second[0])],
            yaxis="y2",
            mode="markers",
            marker={"opacity": 0},
            hoverinfo="skip",
        )
    )
    figure.add_shape(
        type="line",
        x0=0,
        x1=tokens[-1],
        y0=steady,
        y1=steady,
        line={"color": TITLE_COLOR, "width": 1, "dash": "dot"},
    )
    figure.add_annotation(
        value_annotation(
            f"steady {steady / 1e3:.0f}K tokens/s = {steady_mfu:.1f}% MFU",
            x=tokens[len(tokens) // 2],
            y=steady,
            ax=0,
            ay=-46,
        )
    )
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render README telemetry SVGs for a pretrain run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/pretrain-214m-b200/pretrain_214m_b200"),
        help="Run directory containing metrics.jsonl, throughput.csv, run-summary.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        telemetry = load_telemetry(args.run_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "loss_vs_tokens": args.output_dir / "fig-pretrain-loss-vs-tokens.svg",
        "throughput_mfu": args.output_dir / "fig-pretrain-throughput-mfu.svg",
    }
    build_loss_figure(telemetry).write_image(
        outputs["loss_vs_tokens"], format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )
    build_throughput_figure(telemetry).write_image(
        outputs["throughput_mfu"], format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )

    summary = {
        "run_dir": str(args.run_dir),
        "train_points": len(telemetry.train_steps),
        "val_points": len(telemetry.val_steps),
        "throughput_points": len(telemetry.throughput_steps),
        "final_val_loss": telemetry.val_loss[-1],
        "steady_tokens_per_second": statistics.median(telemetry.tokens_per_second),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        for key, path in outputs.items():
            print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
