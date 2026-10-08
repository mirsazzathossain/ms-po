"""Hydra/OmegaConf helpers shared by all pipeline stages."""

from __future__ import annotations

import os

from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


def _basename(name: str) -> str:
    return str(name).rstrip("/").split("/")[-1]


def _ms_suffix(variant: str, gamma: float) -> str:
    """Empty for the paper's default MS-PO setting, otherwise a tag that keeps ablations apart."""
    if variant == "direct" and float(gamma) == 1.0:
        return ""
    return f"_{variant}_g{gamma}"


def register_resolvers() -> None:
    for name, fn in (("basename", _basename), ("ms_suffix", _ms_suffix)):
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


register_resolvers()


def to_container(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)
