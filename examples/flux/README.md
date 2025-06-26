# Chipmunk + FLUX

Original repo by Black Forest Labs: https://blackforestlabs.ai. 


| **Method** | **FLOPs ↓** | **Speedup ↑** | **Latency (s) ↓** | **ImRe ↑** |
|------------|-------------|---------------|-------------------|------------|
| **FLUX.1-dev, T = 50 (768 × 1280)** | | | | |
| Flux                           | 100 % | 1 ×    | 6.60 | 0.76 |
| DiTFastAttn                    | 83 %  | 1.09 × | 6.05 | **0.80** |
| **Chipmunk**                   | **58 %** | **1.41 ×** | **4.90** | **0.80** |
| Step + Token Caching (ToCa)    | 66 %  | 1.51 × | 4.37 | 0.76 |
| Step Caching (TeaCache)        | 39 %  | 2.51 × | 2.64 | 0.68 |
| **Chipmunk + Step Cache**      | **31 %** | **2.56 ×** | **2.57** | **0.77** |

*Performance comparison of various methods on ImageReward (image generation).*

| **Method** | **FLOPs ↓** | **Speedup ↑** | **Latency (s) ↓** | **GenEval ↑** | **CLIP ↑** |
|------------|-------------|---------------|-------------------|---------------|------------|
| **FLUX.1-dev, T = 50 (768 × 1280)** | | | | | |
| Flux                           | 100 % | 1 ×    | 6.60 | 0.66 | 31.07 |
| Step + Token Caching (ToCa)    | 66 %  | 1.51 × | 4.37 | 0.65 | 31.21 |
| Step Caching (TeaCache)        | 45 %  | 2.23 × | 2.95 | 0.61 | 31.37 |
| **Chipmunk-77 % + Step Cache** | **31 %** | **2.56 ×** | **2.57** | 0.62 | 31.18 |
| Chipmunk-65 % + Step Cache     | 38 %  | 2.25 × | 2.93 | **0.66** | **31.43** |

*Performance comparison of various methods across various GenEval and CLIP.*

## Quickstart

### 1\. Clone repo, build kernels, & install deps
Follow the Quickstart instructions in the [base directory](../../README.md) to install Chipmunk's base collection of primitives.

### 2\. Generate fast images!

The first and second images will be slow due to `torch.compile` cold starts. You will speedups for the third image and onwards. For reference, after the first two images, you should see image generation times of ~4.83 seconds per image at a resolution of 1280x768 on the default sparsity config we provide.

```bash
cd <repo_root>/examples/flux

export PROMPT="A very cute cartoon chipmunk dressed up as a ninja holding katanas"
python -m flux.cli --name flux-dev --prompt "$PROMPT" --loop --chipmunk-config ./chipmunk-config.yml
```

For the kontext model, you can run:
~~~
python -m flux.cli_kontext --name flux-dev-kontext \
    --loop \
    --chipmunk-config kontext-chipmunk-config.yml
~~~

### 3\. Play around with sparsity settings

You can edit `chipmunk-config.yml` to your liking! Here are a few parameters that make the most impact on speed:

- **MLP Sparsity:** `mlp.top_keys` - This is one out of the two primary tuning knobs of Chipmunk. `mlp.top_keys` represents for every token group of MLPs, what \% of neurons will be active at once. For example, a value of 0.3 means that 30\% of neurons will be active, and 70\% will be inactive. Our kernels' performance generally scales linearly with the sparsity; you can roughly expect a value of 0.5 to be twice as fast as 0.25. We recommend you experiment with values between 0.3 and 0.7 to strike a good balance between image quality and speed. You can disable MLP sparsity entirely with `mlp.is_enabled: false` and restore default behavior.

- **MLP Full Step Every N Inference Steps:** `mlp.full_step_every` - Chipmunk injects fully dense MLP steps every few inference steps in order to preserve quality. We recommend using values between 5 and 25 for this parameter depending on quality requirements, finding that 10 works well for most use cases.

- **Attention Sparsity:** `attn.top_keys` - This is the second of the two primary tuning knobs of Chipmunk. `attn.top_keys` represents for every query group of attention layers, what \% of keys/values active at once. For example, a value of 0.3 means that every query will attend to 30\% of the total available keys/values. Our kernels' performance generally scales linearly with the sparsity; you can roughly expect a value of 0.5 to be twice as fast as 0.25. Since attention is typically even sparser than MLPs, we've found that you can use values as low as 0.1 (or even 0.05) while preserving image quality. You can disable attention sparsity entirely with `attn.is_enabled: false` and restore default behavior.

- **Attention Full Step Every N Inference Steps:** `attn.full_step_every` - Chipmunk injects fully dense attention steps every few inference steps in order to preserve quality. We recommend using values between 5 and 25 for this parameter depending on quality requirements, finding that 10 works well for most use cases.

- **MLP FP8 (Preview):** Our kernels support FP8. *NOTE: This interacts poorly with torch.compile in the public repository, so while it's faster than BF16, it might not be as fast as other FP8 implementations.*


## Add Chipmunk to your own Flux repo (or inference server!)

We implemented the Chipmunk algorithm on top of the base Flux repo to show you how easy it is to sparsify the MLP and attention layers of any model. Don't worry, our changes are not complex or drastic! In < 20 lines of changes to the `forward()` passes, we were able to sparsify the model. There's a bit of set-up boilerplate.

Want to implement Chipmunk on top of your own Flux repo? We've made a very convenient [diff file](./how-to-add-chipmunk-to-base-flux.patch) that shows exactly what's new in this implementation, so that you can copy and paste the snippets of code into your own Flux inference server!
