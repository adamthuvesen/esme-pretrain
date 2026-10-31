"""Shared Modal container image for pretrain and GPU smoke scripts."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REPO_SRC_MOUNT = "/root/src"
# parents[3] is the repo root locally and /root in the container, where the files are mounted.
REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_VERSION_FILE = REPO_ROOT / ".python-version"
UV_LOCK_FILE = REPO_ROOT / "uv.lock"
LOCAL_PYTHON = PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip()
UV_VERSION = "0.13.5"
IMAGE_PACKAGES = ("torch", "numpy", "pydantic", "datasets", "tokenizers", "wandb")


def locked_version(name: str) -> str:
    """Return the one version of `name` in uv.lock."""
    packages = tomllib.loads(UV_LOCK_FILE.read_text(encoding="utf-8"))["package"]
    versions = {package["version"] for package in packages if package["name"] == name}
    if len(versions) != 1:
        raise ValueError(
            f"{name} must appear in uv.lock exactly once, found {sorted(versions)}. "
            "Add it to pyproject.toml and run `uv lock`."
        )
    return versions.pop()


def image_package_pins() -> dict[str, str]:
    """Return the image packages at their uv.lock versions, so GPU runs match the CPU tests."""
    return {name: locked_version(name) for name in IMAGE_PACKAGES}


def build_pretrain_modal_image(*, repo_root_src: str, modal_module: Any) -> Any:
    """Return a Modal image with locked deps and the local ``src/`` tree mounted."""
    pins = (f"{name}=={version}" for name, version in image_package_pins().items())
    return (
        modal_module.Image.debian_slim(python_version=LOCAL_PYTHON)
        .uv_pip_install(*pins, uv_version=UV_VERSION)
        .env(
            {
                "PYTHONPATH": REPO_SRC_MOUNT,
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        .add_local_dir(repo_root_src, remote_path=REPO_SRC_MOUNT)
        .add_local_file(str(PYTHON_VERSION_FILE), remote_path="/root/.python-version")
        .add_local_file(str(UV_LOCK_FILE), remote_path="/root/uv.lock")
    )
