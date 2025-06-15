# Chipmunk + Mochi

Original Mochi repository: [Mochi](https://github.com/genmoai/mochi)

## Quickstart

### 1. Clone repo, build kernels, & install deps

Follow the Quickstart instructions in the [base Chipmunk directory](https://github.com/sandyresearch/chipmunk/blob/master/README.md) to install Chipmunk’s base collection of primitives.

### 2. Download Mochi Weights

```bash
cd <repo_root>/examples/mochi
python3 ./scripts/download_weights.py weights/
```

### 3. Generate fast videos!

For reference, expect a baseline video generation time of approximately ~380 seconds per video using the default sparsity config provided on a single H100 SXM5 GPU. Chipmunk will accelerate this to ~240 seconds per video (1.4x speedup) without losing quality! Chipmunk manages a pool of pinned CPU memory for efficient just-in-time offloading; initialization may take a few minutes to allocate pinned buffers in RAM.

```bash
./run.sh
```

### 4. Play around with sparsity settings

You can edit `chipmunk-config.yml` to your liking! Here are a few parameters that make the most impact on speed:

- **Attention Sparsity:** `attn.top_keys` - This is the primary tuning knob of Chipmunk in Mochi. `attn.top_keys` represents for every query group of attention layers, what \% of keys/values active at once. For example, a value of 0.3 means that every query will attend to 30\% of the total available keys/values. Our kernels' performance generally scales linearly with the sparsity; you can roughly expect a value of 0.5 to be twice as fast as 0.25. Since attention is typically even sparser than MLPs, we've found that you can use values as low as 0.1 (or even 0.05) while preserving image quality. You can disable attention sparsity entirely with `attn.is_enabled: false` and restore default behavior.

- **Attention Full Step Every N Inference Steps:** `attn.full_step_every` - Chipmunk injects fully dense attention steps every few inference steps in order to preserve quality. We recommend using values between 5 and 25 for this parameter depending on quality requirements, finding that 10 works well for most use cases.

- **Local Voxels:** `local_voxels` - Chipmunk stacks with static sparsity patterns such as local attention. This value will control the size of the local window.
