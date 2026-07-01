"""Shared Modal container image for pretrain and GPU smoke scripts."""

from __future__ import annotations

from typing import Any

from esme_pretrain.launch.pretrain import IMAGE_PACKAGE_PINS

REPO_SRC_MOUNT = "/root/src"


def build_pretrain_modal_image(*, repo_root_src: str, modal_module: Any) -> Any:
    """Return a Modal image with pinned deps and the local ``src/`` tree mounted."""
    return (
        modal_module.Image.debian_slim(python_version="3.11")
        .pip_install(*(f"{name}=={version}" for name, version in IMAGE_PACKAGE_PINS.items()))
        .env(
            {
                "PYTHONPATH": REPO_SRC_MOUNT,
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        .add_local_dir(repo_root_src, remote_path=REPO_SRC_MOUNT)
    )
