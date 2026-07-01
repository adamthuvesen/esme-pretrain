from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy: No module named 'numpy'.*",
    category=UserWarning,
)

import torch  # noqa: E402

__all__ = ["torch"]
