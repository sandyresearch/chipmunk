from __future__ import annotations

"""Batch sampling wrapper for the *HiDream* image model."""

import random
from pathlib import Path
from typing import Any

from chipmunk.evals.batch_sample_main import main as _batch_main

_model: Any | None = None


def init() -> None:
    global _model
    if _model is not None:
        return
    print("[hidream/batch_sample] init() called – model initialisation not yet implemented.")
    _model = object()


def sample(prompt: str, out_file: list[str], seed: int) -> None:
    if _model is None:
        raise RuntimeError("init() must be called before sample().")
    for path in out_file:
        Path(path).write_text(f"Stub (HiDream) {prompt} seed={seed} rnd={random.random()}\n")


if __name__ == "__main__":
    _batch_main(init, sample) 