"""Run metrics logging: durable local record + optional Weights & Biases mirror.

The committed ``metrics.jsonl`` / ``throughput.csv`` are the **durable** record and
are written regardless of W&B. W&B is **optional with graceful fallback**: if the
``wandb`` package is missing, no ``WANDB_API_KEY`` is set, or ``wandb.init`` fails
for any reason, the logger drops to offline/disabled and the run continues — it
never crashes the training loop. The chosen W&B status is surfaced on the logger
(``wandb_status`` / ``wandb_note``) so the run summary can flag an offline fallback.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METRICS_FILE = "metrics.jsonl"
THROUGHPUT_FILE = "throughput.csv"
THROUGHPUT_COLUMNS = ("step", "tokens", "tokens_per_second", "mfu", "step_time_ms")


@dataclass
class WandbSettings:
    enabled: bool = True
    project: str = "esme-pretrain"
    run_name: str | None = None
    run_id: str | None = None
    resume: str | None = None
    entity: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    # Force a mode ("online" | "offline" | "disabled"); None = auto from WANDB_API_KEY.
    mode: str | None = None


class RunLogger:
    """Writes metrics to local files and (optionally) mirrors them to W&B."""

    def __init__(self, output_dir: Path, settings: WandbSettings | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings or WandbSettings()
        self._metrics_path = self.output_dir / METRICS_FILE
        self._throughput_path = self.output_dir / THROUGHPUT_FILE
        self._metrics_handle = self._metrics_path.open("a", encoding="utf-8")
        if not self._throughput_path.exists() or self._throughput_path.stat().st_size == 0:
            self._throughput_path.write_text(",".join(THROUGHPUT_COLUMNS) + "\n", encoding="utf-8")
        self._throughput_handle = self._throughput_path.open("a", encoding="utf-8")

        self.wandb_run: Any = None
        self.wandb_status: str = "disabled"
        self.wandb_note: str | None = None
        self.wandb_run_id: str | None = None
        self.wandb_run_url: str | None = None
        self._init_wandb()

    def _init_wandb(self) -> None:
        if not self.settings.enabled:
            self.wandb_status = "disabled"
            return
        try:
            import wandb
        except (ImportError, OSError) as error:
            self.wandb_status = "unavailable"
            self.wandb_note = f"wandb import failed ({error}); local metrics only"
            return

        mode = self.settings.mode
        if mode is None and not os.environ.get("WANDB_API_KEY"):
            # No key -> offline so a missing secret never blocks or prompts.
            mode = "offline"
        try:
            self.wandb_run = wandb.init(
                project=self.settings.project,
                name=self.settings.run_name,
                id=self.settings.run_id,
                resume=self.settings.resume,
                entity=self.settings.entity,
                config=self.settings.config,
                tags=self.settings.tags or None,
                mode=mode,
                dir=str(self.output_dir),
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.wandb_run = None
            self.wandb_status = "unavailable"
            self.wandb_note = f"wandb.init failed ({error}); local metrics only"
            return

        self.wandb_status = "offline" if mode == "offline" else (mode or "online")
        self.wandb_run_id = getattr(self.wandb_run, "id", None)
        # `.url` is the current API (get_url() is deprecated); offline runs have none.
        self.wandb_run_url = getattr(self.wandb_run, "url", None)
        if self.wandb_run_id:
            (self.output_dir / "wandb-run-id.txt").write_text(
                f"{self.wandb_run_id}\n", encoding="utf-8"
            )

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Append one step's metrics to metrics.jsonl (+ throughput.csv) and W&B."""
        record = {"step": int(step), **metrics}
        self._metrics_handle.write(json.dumps(record) + "\n")
        self._metrics_handle.flush()
        if "tokens_per_second" in metrics:
            row = [
                str(step),
                str(metrics.get("tokens", "")),
                f"{metrics['tokens_per_second']:.3f}",
                f"{metrics['mfu']:.5f}" if metrics.get("mfu") is not None else "",
                f"{metrics['step_time_ms']:.3f}" if metrics.get("step_time_ms") is not None else "",
            ]
            self._throughput_handle.write(",".join(row) + "\n")
            self._throughput_handle.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def log_samples(self, step: int, prompt: str, samples: list[str]) -> None:
        """Record a couple of qualitative generations (local file + W&B)."""
        path = self.output_dir / "samples.md"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## step {step}\n\nprompt: {prompt!r}\n\n")
            for index, text in enumerate(samples):
                handle.write(f"- sample {index}: {text!r}\n")
        if self.wandb_run is not None:
            with contextlib.suppress(ImportError, OSError, RuntimeError, ValueError):
                import wandb

                table = wandb.Table(columns=["step", "prompt", "sample"])
                for text in samples:
                    table.add_data(step, prompt, text)
                self.wandb_run.log({"samples": table}, step=step)

    def summary(self, values: dict[str, Any]) -> None:
        (self.output_dir / "run-summary.json").write_text(
            json.dumps(values, indent=2), encoding="utf-8"
        )
        if self.wandb_run is not None:
            self.wandb_run.summary.update(values)

    def finish(self) -> None:
        self._metrics_handle.close()
        self._throughput_handle.close()
        if self.wandb_run is not None:
            with contextlib.suppress(
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                ValueError,
            ):
                self.wandb_run.finish()
