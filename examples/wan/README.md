# Chipmunk + Wan2.1

Original repo: https://github.com/Wan-Video/Wan2.1

https://github.com/user-attachments/assets/4a9af692-0926-49a6-9d00-a975a383d209

## Quickstart

### 1. Clone repo, build kernels, & install deps

Follow the Quickstart instructions in the [base Chipmunk directory](../../README.md) to install Chipmunk's base collection of primitives.

### 2\. Download Wan2.1 Weights

```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

### 3\. Generate fast videos!

For reference, you should see video generation times of ~510s seconds per video on the default sparsity config we provide. Because of Chipmunk's just-in-time offloading, we manage a pool of pinned CPU memory. Model initialization may take up to ~5 minutes as we allocate all these pinned buffers in RAM, but you can interactively enter prompts after this without re-initializing!

```bash
cd <repo_root>/examples/wan
./run.sh
```

### 3\. Play around with sparsity settings

You can edit `chipmunk-config.yml` to your liking! Here are a few parameters that make the most impact on speed:

- **Attention Sparsity:** `attn.top_keys` - This is the primary tuning knob of Chipmunk. `attn.top_keys` represents for every query group of attention layers, what \% of keys/values active at once. For example, a value of 0.3 means that every query will attend to 30\% of the total available keys/values. Our kernels' performance generally scales linearly with the sparsity; you can roughly expect a value of 0.5 to be twice as fast as 0.25. Since attention is typically even sparser than MLPs, we've found that you can use values as low as 0.1 (or even 0.05) while preserving image quality. You can disable attention sparsity entirely with `attn.is_enabled: false` and restore default behavior.

- **Attention Full Step Every N Inference Steps:** `attn.full_step_every` - Chipmunk injects fully dense attention steps every few inference steps in order to preserve quality. We recommend using values between 5 and 25 for this parameter depending on quality requirements, finding that 10 works well for most use cases.

- **Local Voxels:** `local_voxels` - Chipmunk stacks with static sparsity patterns such as local attention. This value will control the size of the local window.
