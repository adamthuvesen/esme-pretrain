#!/usr/bin/env python3
from __future__ import annotations

from esme_pretrain.launch import gpu_smoke as _gpu_smoke

launch = _gpu_smoke.launch
if hasattr(_gpu_smoke, "app"):  # pragma: no cover - defined only when Modal is installed.
    app = _gpu_smoke.app
    main = _gpu_smoke.main
    run_smoke = _gpu_smoke.run_smoke


if __name__ == "__main__":
    raise SystemExit(launch())
