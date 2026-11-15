#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

from esme_pretrain.launch.modal_image import build_pretrain_modal_image
from esme_pretrain.launch.modal_pretrain_body import (
    APP_NAME,
    REPO_ROOT,
    VOLUME_MOUNT,
    launch,
    run_pretrain_launch_body,
)

PRETRAIN_GPU = os.environ.get("PRETRAIN_GPU", "H100")
PRETRAIN_TIMEOUT_HOURS = int(float(os.environ.get("PRETRAIN_TIMEOUT_HOURS", "19")))
if PRETRAIN_TIMEOUT_HOURS > 24:
    raise ValueError("PRETRAIN_TIMEOUT_HOURS must not exceed Modal's 24h function maximum")

try:
    import modal
except ImportError:  # pragma: no cover - Modal is supplied by the launch command.
    modal = None

app: Any = None
run_pretrain_launch = None

if modal is not None:  # pragma: no cover - exercised by Modal, not local unit tests.
    image = build_pretrain_modal_image(
        repo_root_src=str(REPO_ROOT / "src"),
        modal_module=modal,
    )
    pretrain_volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)
    app = modal.App(APP_NAME, image=image)

    @app.function(
        gpu=PRETRAIN_GPU,
        # Keep the Modal hard timeout aligned with the configured runtime spend stop.
        timeout=PRETRAIN_TIMEOUT_HOURS * 60 * 60,
        volumes={str(VOLUME_MOUNT): pretrain_volume},
        secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    )
    def run_pretrain_launch(
        config_payload: dict[str, Any], commit: str, dirty: bool
    ) -> dict[str, Any]:
        try:
            return run_pretrain_launch_body(config_payload, commit=commit, dirty=dirty)
        finally:
            # Persist intermediate checkpoints on controlled aborts. A hard Modal
            # kill can still interrupt anything, so the body also arms a softer alarm.
            pretrain_volume.commit()


if __name__ == "__main__":
    raise SystemExit(
        launch(
            app=app,
            run_pretrain_launch=run_pretrain_launch,
            modal_gpu=PRETRAIN_GPU,
            modal_timeout_hours=PRETRAIN_TIMEOUT_HOURS,
        )
    )
