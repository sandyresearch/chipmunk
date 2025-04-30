# batch_sample_ray.py
from __future__ import annotations
import os, sys, pickle, traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import ray

from chipmunk.evals.batch_sample_main import main as _batch_main

# --------------------------------------------------------------------------- #
#                           model-level helpers                               #
# --------------------------------------------------------------------------- #

_model: Any | None = None     # holds HunyuanVideoSampler
_global_args: Any | None = None


def init() -> None:
    """Initialised once per Ray actor *after* dist PG is ready."""
    global _model, _global_args
    if _model is not None:           # already done on this rank
        return

    # ── sync all workers *before* heavy initialisation ──
    if dist.is_initialized():
        dist.barrier()

    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler

    # parse Hunyuan CLI with defaults only
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    try:
        _global_args = parse_args()
    finally:
        sys.argv = old_argv
    _global_args.flow_reverse = True                 # user default

    device = torch.device("cuda")
    models_root = Path(_global_args.model_base)
    if not models_root.exists():
        raise FileNotFoundError(f"Model base not found: {models_root}")

    _model = HunyuanVideoSampler.from_pretrained(
        models_root, args=_global_args, device=device
    )
    _model.pipeline.transformer.sparsify()

    # ── optional sync so that all ranks finish loading together ──
    dist.barrier()


def sample(prompt: str, out_file: list[str], seed: int) -> None:
    """Called repeatedly by the harness on each rank."""
    if _model is None:
        raise RuntimeError("init() must be called first")
    from hyvideo.utils.file_utils import save_videos_grid
    dist.barrier()

    outs = _model.predict(
        prompt                  = prompt,
        height                  = _global_args.video_size[0],
        width                   = _global_args.video_size[1],
        video_length            = _global_args.video_length,
        seed                    = seed,
        negative_prompt         = _global_args.neg_prompt,
        infer_steps             = _global_args.infer_steps,
        guidance_scale          = _global_args.cfg_scale,
        num_videos_per_prompt   = 1,
        flow_shift              = _global_args.flow_shift,
        batch_size              = _global_args.batch_size,
        embedded_guidance_scale = _global_args.embedded_cfg_scale,
    )

    video = outs["samples"][0].unsqueeze(0)
    for p in out_file:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        save_videos_grid(video, p, fps=24)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    dist.barrier()
    
    return outs["gen_time"]

# --------------------------------------------------------------------------- #
#                     Ray actor wrapping the harness                          #
# --------------------------------------------------------------------------- #

@ray.remote(num_gpus=1)
def _ray_worker(argv_pickled: bytes, local_rank: int, world_size: int) -> None:
    """Single-GPU worker: set up NCCL then run the harness loop."""
    # re-inject CLI args the outer user supplied
    sys.argv = pickle.loads(argv_pickled)

    # environment for NCCL
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"]       = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="nccl",
        rank=local_rank,
        world_size=world_size,
    )

    # Hunyuan’s head-parallel helper (same as in first file)
    from hyvideo.modules.head_parallel import setup_dist
    setup_dist(dist.group.WORLD, local_rank, world_size)

    # start the evaluation harness – this calls our init()/sample()
    _batch_main(init, sample)

# --------------------------------------------------------------------------- #
#                       driver (mirrors first script)                         #
# --------------------------------------------------------------------------- #

def _spawn_all() -> None:
    """Launch one Ray actor per visible GPU and wait for completion."""
    # keep user CLI flags for the workers
    argv_blob = pickle.dumps(sys.argv)

    world = torch.cuda.device_count() or 1

    actors = [
        _ray_worker.remote(argv_blob, i, world)
        for i in range(world)
    ]
    # block until all actors finish (or raise)
    try:
        ray.get(actors)
    except Exception:
        traceback.print_exc()
        for a in actors:
            ray.kill(a, no_restart=True)
        raise


if __name__ == "__main__":
    import chipmunk.util.config
    chipmunk.util.config.load_from_file("chipmunk-config.yml")

    ray.init(_temp_dir="/tmp/ray-hunyuan-batch", ignore_reinit_error=True)
    _spawn_all()