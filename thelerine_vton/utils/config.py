"""
Configuration utilities for ThelerineVTON.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config