"""Dataset code. Each loader returns `datasets.Dataset` splits with string columns
`prompt`, `chosen`, `rejected` (human preference labels)."""

from dataset.registry import build_splits

__all__ = ["build_splits"]
