"""Dataset loaders returning `prompt`, `chosen`, `rejected` splits."""

from dataset.registry import build_splits

__all__ = ["build_splits"]
