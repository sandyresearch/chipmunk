from __future__ import annotations
from chipmunk.evals.batch_sample_main import main as _batch_main

"""Batch sampling wrapper for the *Flux* model family.

This file is **auto-generated**.  Replace the body of `init()` and `sample()`
with the logic from the reference inference script (see examples/flux/README.md)
so that the model is initialised *once* and subsequent calls to ``sample`` reuse
that state.
"""

import torch
torch._dynamo.config.cache_size_limit = 1 << 31
import random
from pathlib import Path
from typing import Any

from chipmunk.util import GLOBAL_CONFIG
# -----------------------------------------------------------------------------
# Globals set during `init()`
# -----------------------------------------------------------------------------
# The heavy model components are loaded exactly once by `init()` and reused by
# successive calls to `sample()` within the same Python process (this is the
# behaviour expected by the Chipmunk batch-sampling harness).

_model: Any | None = None  # Flux flow model
_t5 = None  # T5 text encoder
_clip = None  # CLIP text/vision encoder
_ae = None  # Auto-encoder
_device = None  # torch.device used for inference

# Default hyper-parameters – feel free to tweak if you know what you're doing.
_MODEL_NAME = "flux-dev"  # Smallest, fastest variant
_GUIDANCE = 3.5  # Same default as the CLI script

# -----------------------------------------------------------------------------
# Public API expected by the harness
# -----------------------------------------------------------------------------

def init() -> None:  # noqa: D401
    """Initialise the heavy model weights *once* per Python process.

    The evaluation harness will call :func:`init` exactly once on each process
    (or rank).  All subsequent calls to :func:`sample` are expected to reuse
    the state created here – this massively speeds up batched evaluations.
    """
    global _model, _t5, _clip, _ae, _device

    if _model is not None:
        # Already initialised – nothing to do.
        return

    import torch
    from flux.util import load_t5, load_clip, load_flow_model, load_ae

    # ---------------------------------------------------------------------
    # Select device (prefer GPU when available)
    # ---------------------------------------------------------------------
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # Load individual components
    # ---------------------------------------------------------------------
    _t5 = load_t5(_device, max_length=256 if _MODEL_NAME == "flux-schnell" else 512)
    _clip = load_clip(_device)
    _model = load_flow_model(_MODEL_NAME, device=_device)
    _ae = load_ae(_MODEL_NAME, device=_device)

    print(f"[flux/batch_sample] Model '{_MODEL_NAME}' initialised on {_device}.")


@torch.inference_mode()
def sample(prompt: str, out_file: str, seed: int) -> None:  # noqa: D401
    """Generate an image for *prompt* and save it to *out_file*.

    The implementation follows the logic from the reference CLI (``flux/cli.py``)
    but omits interactive functionality and other extras that are not required
    for the Chipmunk evaluation harness.
    """
    H = 128 * (GLOBAL_CONFIG["height"] // 128)   # 2️⃣ match CLI logic
    W = 128 * (GLOBAL_CONFIG["width"]  // 128)

    if _model is None:
        raise RuntimeError("init() must be called before sample().")

    import torch
    from einops import rearrange
    from PIL import Image

    from flux.sampling import (
        denoise,
        get_noise,
        get_schedule,
        prepare,
        unpack,
    )

    # ---------------------------------------------------------------------
    # Deterministic seeding
    # ---------------------------------------------------------------------
    if seed == 0:
        # A seed of zero in the prompt list means "no specific seed" – emulate
        # the CLI behaviour by picking a random one.
        seed = random.randrange(1, 2**31 - 1)

    # Ensure output directory exists
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    torch_device = _device

    # ---------------------------------------------------------------------
    # Prepare latent noise
    # ---------------------------------------------------------------------
    x = get_noise(
        1,
        H,
        W,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    inp = prepare(_t5, _clip, x, prompt)

    timesteps = get_schedule(
        GLOBAL_CONFIG['steps'],
        inp["img"].shape[1],
        shift=(_MODEL_NAME != "flux-schnell"),
    )

    # ---------------------------------------------------------------------
    # Denoising
    # ---------------------------------------------------------------------
    x = denoise(
        _model,
        timesteps=timesteps,
        guidance=_GUIDANCE,
        **inp,
    )

    # ---------------------------------------------------------------------
    # Decode latents to RGB space
    # ---------------------------------------------------------------------
    x = unpack(x.float(), H, W)

    if torch_device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = _ae.decode(x)
    else:
        x = _ae.decode(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ---------------------------------------------------------------------
    # Convert to PIL Image & save
    # ---------------------------------------------------------------------
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(out_file, quality=95, subsampling=0)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    _batch_main(init, sample) 