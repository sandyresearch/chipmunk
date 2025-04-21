# Chipmunk + HunyuanVideo

Original repo: https://github.com/Tencent/HunyuanVideo



https://github.com/user-attachments/assets/b68f5c08-6acc-4915-99a8-e80293836aea



## Quickstart
### 1\. Download Weights
* Download the HunyuanVideo weights from here: https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md 
* Save the weights to `./ckpts/`

### 2\. Clone repo, build kernels, & install deps
Follow the Quickstart instructions in the [base directory](../../README.md) to install Chipmunk's base collection of primitives.

### 3\. Generate fast videos!

The first video will be slightly slower due to `torch.compile` cold starts (by about 10s). For reference, you should see video generation times of ~280s seconds per video at a resolution of 720x1280 on the default sparsity config we provide.

```bash
cd <repo_root>/examples/hunyuan

python3 sample_video.py --flow-reverse --chipmunk-config ./chipmunk-config.yml
```

### 3\. Play around with sparsity settings

You can edit `chipmunk-config.yml` to your liking! Here are a few parameters that make the most impact on speed:

- **Attention Sparsity:** `attn.top_keys` - This is the primary tuning knob of Chipmunk in HunyuanVideo. `attn.top_keys` represents for every query group of attention layers, what \% of keys/values active at once. For example, a value of 0.3 means that every query will attend to 30\% of the total available keys/values. Our kernels' performance generally scales linearly with the sparsity; you can roughly expect a value of 0.5 to be twice as fast as 0.25. Since attention is typically even sparser than MLPs, we've found that you can use values as low as 0.1 (or even 0.05) while preserving image quality. You can disable attention sparsity entirely with `attn.is_enabled: false` and restore default behavior.

- **Attention Full Step Every N Inference Steps:** `attn.full_step_every` - Chipmunk injects fully dense attention steps every few inference steps in order to preserve quality. We recommend using values between 5 and 25 for this parameter depending on quality requirements, finding that 10 works well for most use cases.
