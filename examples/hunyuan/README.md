# Chipmunk + HunyuanVideo

Original repo: https://github.com/Tencent/HunyuanVideo



https://github.com/user-attachments/assets/b68f5c08-6acc-4915-99a8-e80293836aea



## Quickstart
### 1. Clone repo, build kernels, & install deps
Follow the Quickstart instructions in the [base Chipmunk directory](../../README.md) to install Chipmunk's base collection of primitives.

### 2\. Download Hunyuan Weights

```bash
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava-llama-3-8b-v1_1-transformers
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ./ckpts/text_encoder
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2
```

### 3\. Generate fast videos!

The first video will be slightly slower due to `torch.compile` cold starts (by about 10s). For reference, you should see video generation times of ~280s seconds per video at a resolution of 720x1280 on the default sparsity config we provide. Additionally, because of Chipmunk's just-in-time offloading, we manage a pool of pinned CPU memory. Model initialization may take up to ~5 minutes as we allocate all these pinned buffers in RAM!

```bash
cd <repo_root>/examples/hunyuan
python3 sample_video.py --flow-reverse --chipmunk-config ./chipmunk-config.yml
```

### 3\. Play around with sparsity settings

You can edit `chipmunk-config.yml` to your liking! Here are a few parameters that make the most impact on speed:

- **Attention Sparsity:** `attn.top_keys` - This is the primary tuning knob of Chipmunk in HunyuanVideo. `attn.top_keys` represents for every query group of attention layers, what \% of keys/values active at once. For example, a value of 0.3 means that every query will attend to 30\% of the total available keys/values. Our kernels' performance generally scales linearly with the sparsity; you can roughly expect a value of 0.5 to be twice as fast as 0.25. Since attention is typically even sparser than MLPs, we've found that you can use values as low as 0.1 (or even 0.05) while preserving image quality. You can disable attention sparsity entirely with `attn.is_enabled: false` and restore default behavior.

- **Attention Full Step Every N Inference Steps:** `attn.full_step_every` - Chipmunk injects fully dense attention steps every few inference steps in order to preserve quality. We recommend using values between 5 and 25 for this parameter depending on quality requirements, finding that 10 works well for most use cases.

Since video generation models are dominated by attention (attention takes ~10x the amount of time as the MLP), we didn't even need MLP sparsity to see massive speedups!