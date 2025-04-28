from __future__ import annotations

"""Batch sampling wrapper for the *HunyuanVideo* model family.

Copy the logic from ``examples/hunyuan/sample_video.py`` such that the heavy
initialisation happens once in :func:`init`, while :func:`sample` performs the
per-prompt generation.
"""

import random
from pathlib import Path
from typing import Any

from chipmunk.evals.batch_sample_main import main as _batch_main

_model: Any | None = None  # will hold the sampler object
_global_args: Any | None = None  # will hold parsed arguments for reuse


def init() -> None:
    """Initialise the Hunyuan model (stub)."""
    global _model, _global_args
    if _model is not None:
        return
    # TODO: import and reuse HunyuanVideoSampler initialisation from sample_video.py
    import sys
    from pathlib import Path

    import torch
    import ray

    # We import lazily to avoid the heavy dependency cost when this module is merely
    # inspected (e.g. by static analysis tools).
    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler

    # The evaluation harness adds its own CLI flags (``--prompt-file`` and
    # ``--chipmunk-config``).  These are not understood by Hunyuan's
    # ``parse_args`` helper and would cause a failure.  We therefore create a
    # temporary copy of ``sys.argv`` that contains *only* the program name so
    # that all model-specific options fall back to their default values.
    old_argv = sys.argv
    sys.argv = [old_argv[0]]  # preserve script name, drop the rest
    try:
        _global_args = parse_args()  # use built-in defaults
    finally:
        sys.argv = old_argv  # restore original argv for the outer harness

    # The user requested to assume ``--flow_reverse`` is enabled by default.
    # The flag is a boolean that defaults to *False*, so we flip it here.
    _global_args.flow_reverse = True

    # Single-GPU setup: pick the first CUDA device when available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional: initialise Ray so that underlying helpers expecting it work
    # even when we only use a single GPU.
    if not ray.is_initialized():
        ray.init(_temp_dir="/tmp/ray-hunyuan-batch")

    models_root_path = Path(_global_args.model_base)
    if not models_root_path.exists():
        raise FileNotFoundError(f"Model base path not found: {models_root_path}")

    # Heavy-weight initialisation – this may take a while.
    print("[hunyuan/batch_sample] Loading HunyuanVideo model …")
    _model = HunyuanVideoSampler.from_pretrained(models_root_path, args=_global_args, device=device)
    print("[hunyuan/batch_sample] Model initialised and ready.")


def sample(prompt: str, out_file: str, seed: int) -> None:
    if _model is None:
        raise RuntimeError("init() must be called before sample().")
    import torch
    from hyvideo.utils.file_utils import save_videos_grid

    # Ensure output directory exists.
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    # Generate video for the given prompt.
    outputs = _model.predict(
        prompt=prompt,
        height=_global_args.video_size[0],
        width=_global_args.video_size[1],
        video_length=_global_args.video_length,
        seed=seed,
        negative_prompt=_global_args.neg_prompt,
        infer_steps=_global_args.infer_steps,
        guidance_scale=_global_args.cfg_scale,
        num_videos_per_prompt=1,
        flow_shift=_global_args.flow_shift,
        batch_size=_global_args.batch_size,
        embedded_guidance_scale=_global_args.embedded_cfg_scale,
    )

    samples = outputs["samples"]
    if not samples:
        raise RuntimeError("Model returned no samples.")

    # The sampler returns a list of tensors with shape (C, T, H, W).  Save the
    # first sample as an MP4 using the helper from the reference script.
    video_tensor = samples[0].unsqueeze(0)  # add batch dimension expected by saver
    save_videos_grid(video_tensor, out_file, fps=24)

    # Optional synchronisation to ensure all GPU work is complete before we
    # return (useful for accurate timing in the harness).
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    _batch_main(init, sample) 