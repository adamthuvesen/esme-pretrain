from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
