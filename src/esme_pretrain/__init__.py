"""CPU-first base-model pretraining lab."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("esme-pretrain")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
