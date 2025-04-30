from __future__ import annotations

"""Batch sampling wrapper for the *Mochi* video model.

Replace :func:`init` and :func:`sample` with logic from the reference CLI/API in
``examples/mochi``.
"""

import random
from pathlib import Path
from typing import Any

from chipmunk.evals.batch_sample_main import main as _batch_main

_model: Any | None = None


def init() -> None:
    """Initialise the Mochi video generation pipeline (heavy operation).

    The initialisation logic is adapted from ``examples/mochi/demos/cli.py``.  We
    load the text encoder, DIT, and decoder **once** per Python process and
    reuse the resulting pipeline for all subsequent calls to :func:`sample`.
    """

    global _model

    # Already initialised?  Nothing to do.
    if _model is not None:
        return

    import os
    import torch

    # ------------------------------------------------------------------
    # Locate model weights – by default we use the local ``weights`` folder
    # that comes with the Mochi example.  Alternatively, the user can set the
    # ``MOCHI_MODEL_DIR`` environment variable to point elsewhere.
    # ------------------------------------------------------------------
    from pathlib import Path

    model_dir = Path(os.environ.get("MOCHI_MODEL_DIR", Path(__file__).parent / "weights")).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Mochi weights directory not found: {model_dir}")

    # ------------------------------------------------------------------
    # Create the Mochi pipeline – single-GPU when only one CUDA device is
    # visible, otherwise fall back to the built-in multi-GPU implementation.
    # ------------------------------------------------------------------
    from genmo.mochi_preview.pipelines import (
        DecoderModelFactory,
        DitModelFactory,
        MochiMultiGPUPipeline,
        MochiSingleGPUPipeline,
        T5ModelFactory,
    )

    num_gpus = torch.cuda.device_count()
    print(f"[mochi/batch_sample] Initialising pipeline on {num_gpus} GPU(s)…")

    pipeline_cls = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline

    common_kwargs = dict(
        text_encoder_factory=T5ModelFactory(),
        dit_factory=DitModelFactory(
            model_path=str(model_dir / "dit.safetensors"),
            model_dtype="bf16",
            lora_path=None,
        ),
        decoder_factory=DecoderModelFactory(model_path=str(model_dir / "decoder.safetensors")),
    )

    if num_gpus > 1:
        # Multi-GPU mode currently does not support CPU off-loading or LoRA.
        common_kwargs["world_size"] = num_gpus
        _model = pipeline_cls(**common_kwargs)  # type: ignore[arg-type]
    else:
        common_kwargs.update(
            cpu_offload=True,  # As per user request
            decode_type="tiled_spatial",
            fast_init=True,
            strict_load=True,
            decode_args=dict(overlap=8),
        )
        _model = pipeline_cls(**common_kwargs)  # type: ignore[arg-type]

    print("[mochi/batch_sample] Pipeline initialised – ready to sample.")


def sample(prompt: str, out_file: list[str], seed: int) -> None:
    """Generate a video for *prompt* and save it to *out_file*.

    The generation parameters mirror the defaults used by the reference CLI
    (see ``examples/mochi/demos/cli.py``) with conservative settings that fit
    on a single H100 when *cpu_offload* is enabled.
    """

    if _model is None:
        raise RuntimeError("init() must be called before sample().")

    import os
    import time

    import numpy as np
    from genmo.lib.progress import progress_bar
    from genmo.lib.utils import save_video
    from genmo.mochi_preview.pipelines import linear_quadratic_schedule

    # ------------------------------------------------------------------
    # Fixed generation hyper-parameters (matches CLI defaults)
    # ------------------------------------------------------------------
    HEIGHT = 480
    WIDTH = 848
    NUM_FRAMES = 163
    CFG_SCALE = 6.0
    NUM_STEPS = 64
    THRESHOLD_NOISE = 0.025
    LINEAR_STEPS = None  # falls back to quadratic schedule past threshold

    # Schedule preparation (identical to CLI implementation)
    sigma_schedule = linear_quadratic_schedule(NUM_STEPS, THRESHOLD_NOISE, LINEAR_STEPS)
    cfg_schedule = [CFG_SCALE] * NUM_STEPS

    args = dict(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        sigma_schedule=sigma_schedule,
        cfg_schedule=cfg_schedule,
        num_inference_steps=NUM_STEPS,
        batch_cfg=False,
        prompt=prompt,
        negative_prompt="",
        seed=seed,
    )

    # Ensure output directory exists
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run the pipeline and save the resulting video
    # ------------------------------------------------------------------
    with progress_bar(type="tqdm", enabled=(os.environ.get("RANK", "0") == "0")):
        frames = _model(**args)

        # Pipeline returns a NumPy array with shape (B, T, H, W, 3).
        frames = frames[0]

        assert isinstance(frames, np.ndarray) and frames.dtype == np.float32, "Unexpected output format from pipeline"

        save_video(frames, str(out_path))

    # Small sleep to make sure the video is flushed to disk before returning.
    time.sleep(0.1)


if __name__ == "__main__":
    _batch_main(init, sample) 