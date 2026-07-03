"""Trainer error types."""

from __future__ import annotations


class TrainerError(ValueError):
    """A trainer config or runtime request the training loop cannot honor."""
