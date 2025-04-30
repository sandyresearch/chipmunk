from __future__ import annotations

"""Batch sampling wrapper for the *Wan* video model."""

import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import wan  # type: ignore
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video

from chipmunk.evals.batch_sample_main import main as _batch_main

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

_model: Any | None = None
_cfg = WAN_CONFIGS["t2v-14B"]
_MODEL_ARGS = {
    "config": _cfg,
    "checkpoint_dir": "./Wan2.1-T2V-14B",
    "t5_fsdp": False,
    "dit_fsdp": False,
    "use_usp": False,
    "t5_cpu": False,
}

# Generation parameters (fixed as per prompt)
_GEN_KWARGS = {
    "size": SIZE_CONFIGS["1280*720"],
    "frame_num": 81,  # default for t2v
    "shift": 5.0,
    "sample_solver": "unipc",
    "sampling_steps": 50,  # default in generate.py for t2v
    "guide_scale": 5.0,
    "offload_model": True,
}


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


# -----------------------------------------------------------------------------
# Public API expected by `batch_sample_main`
# -----------------------------------------------------------------------------


def init() -> None:  # noqa: D401
    """Initialise the global Wan model (one per process)."""
    global _model
    if _model is not None:
        return  # Already initialised

    # ------------------------------------------------------------------
    # Distributed setup (mirrors logic in examples/wan/generate.py)
    # ------------------------------------------------------------------
    rank = int(os.getenv("RANK", "0"))
    world_size = _get_world_size()
    local_rank = _get_local_rank()

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
        device_id = local_rank
    else:
        device_id = 0

    # ------------------------------------------------------------------
    # Create the WanT2V pipeline
    # ------------------------------------------------------------------
    print(f"[wan/batch_sample] Initialising WanT2V on device {device_id} (rank {rank})")
    _model = wan.WanT2V(device_id=device_id, rank=rank, **_MODEL_ARGS)  # type: ignore[arg-type]



def sample(prompt: str, out_file: list[str], seed: int) -> None:  # noqa: D401
    """Generate a video for *prompt* and save it to *out_file*."""
    if _model is None:
        raise RuntimeError("init() must be called before sample().")

    # The WanT2V.generate method expects a negative seed of -1 to mean random.
    # We pass the provided seed directly.
    video = _model.generate(prompt, seed=seed, **_GEN_KWARGS)

    # In multi-process runs, only rank 0 returns a tensor; other ranks get None.
    if video is None:
        return  # Nothing to save on this rank

    # Ensure parent directory exists
    for path in out_file:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache_video(
            tensor=video[None],  # add batch dim expected by cache_video
            save_file=str(path),
            fps=_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )


if __name__ == "__main__":
    _batch_main(init, sample) 