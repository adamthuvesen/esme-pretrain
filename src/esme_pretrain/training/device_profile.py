"""GPU peak TFLOP/s lookup for MFU reporting."""

from __future__ import annotations

# bf16 dense (no-sparsity) peak TFLOP/s per GPU, the MFU denominator.
# Substring match, most specific first.
DEVICE_PEAK_TFLOPS: dict[str, float] = {
    "B200": 2250.0,
    "A100": 312.0,
    "H200": 989.0,
    "H100": 989.0,
    "L40S": 362.0,
    "A10G": 125.0,
    "A10": 125.0,
    "L4": 121.0,
    "T4": 65.0,
}


def peak_tflops_for_device(device_name: str) -> float | None:
    for key, value in DEVICE_PEAK_TFLOPS.items():
        if key in device_name:
            return value
    return None
