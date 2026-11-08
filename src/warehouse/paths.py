"""Resolve where training artifacts (models, TensorBoard logs) are written.

By default everything lands under the current working directory, so running the
CLI from the repo root creates ``./models`` and ``./logs``. Override the base
with the ``WAREHOUSE_OUTPUT_DIR`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path


def output_base() -> Path:
    return Path(os.environ.get("WAREHOUSE_OUTPUT_DIR", Path.cwd()))


def models_dir(*parts: str) -> Path:
    p = output_base() / "models"
    for part in parts:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir(*parts: str) -> Path:
    p = output_base() / "logs"
    for part in parts:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p
