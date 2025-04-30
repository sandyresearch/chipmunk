from __future__ import annotations
import os

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Dict, Any
import torch.distributed as dist
import torch
# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------

def _get_world_size() -> int:
    return int(os.environ.get("CHIPMUNK_WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))


def _get_local_rank() -> int:
    # Torchrun sets LOCAL_RANK.  If running inside SLURM, you may only have RANK.
    return int(os.environ.get("CHIPMUNK_LOCAL_RANK", os.environ.get("RANK", "0")))


# -----------------------------------------------------------------------------
# Main procedure
# -----------------------------------------------------------------------------

def main(init_fn: Callable[[], None], sample_fn: Callable[[str, list[str], int], None]) -> None:  # noqa: D401,E501
    parser = argparse.ArgumentParser(description="Chipmunk batch sampling entrypoint")
    parser.add_argument("--prompt-file", required=True, help="JSON prompt file path")
    parser.add_argument("--chipmunk-config", required=True, help="Chipmunk config YAML path")
    args = parser.parse_args()

    cfg_path = Path(args.chipmunk_config).expanduser().resolve()
    prompt_path = Path(args.prompt_file).expanduser().resolve()

    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")
    if not prompt_path.exists():
        sys.exit(f"Prompt file not found: {prompt_path}")
    
    import chipmunk.util.config as _cm_cfg
    _cm_cfg.load_from_file(str(cfg_path))

    with prompt_path.open("r") as f:    
        prompts: List[Dict[str, Any]] = json.load(f)

    # Experiment directory is the parent of the config file
    exp_dir = cfg_path.parent
    media_dir = exp_dir / "media"
    media_dir.mkdir(exist_ok=True)

    world_size = _get_world_size()
    local_rank = _get_local_rank()

    print(f"[batch_sample_main] RANK {local_rank}: using GPU {torch.cuda.current_device()}")

    # Initialize model once per process
    print(f"[batch_sample_main] RANK {local_rank}: initializing model")
    init_fn()

    for idx, item in enumerate(prompts):
        if idx % world_size != local_rank:
            continue

        prompt: str = item["prompt"]
        seed: int = int(item.get("seed", 0))
        out_name: str = item.get("output_path", f"{idx:06d}.png")
        if type(out_name) == str:
            out_name = [out_name]
        out_path = []
        for name in out_name:
            out_path.append(media_dir / name)
        if all(path.exists() for path in out_path):
            continue

        # Ensure deterministic ordering between processes
        print(f"[batch_sample_main] RANK {local_rank}: generating {[x.name for x in out_path]}")
        start_t = time.time()
        try:
            dur = sample_fn(prompt, [str(x) for x in out_path], seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[batch_sample_main] ERROR while sampling {out_path}: {exc}")
            raise
        if dur is None:
            dur = time.time() - start_t
        print(f"[batch_sample_main] RANK {local_rank}: finished in {dur:.2f}s")

    # Only primary rank writes done signal
    if local_rank == 0:
        done_file = exp_dir / "done.txt"
        done_file.touch()


# The file is intended to be imported by per-model wrappers, not executed directly.
if __name__ == "__main__":  # pragma: no cover
    sys.exit("Please import this module and call main(init_fn, sample_fn)") 