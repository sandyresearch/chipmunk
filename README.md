<p align="center">
<h1 align="center">Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas</h1>
</p>
<p align="center">
  <p align="center">
    <a href="https://x.com/austinsilveria">Austin Silveria</a><sup>1,3</sup>
    ·
    <a href="https://sohamgovande.com/">Soham Govande</a><sup>2</sup>
    ·
    <a href="https://danfu.org/">Dan Fu</a><sup>1-3</sup><br/>
        <sup>1</sup>Together AI <sup>2</sup>Stanford University <sup>3</sup>UCSD
  </p>
  <h3 align="center">Accepted to EsFoMo@ICML2025 and YPS@MLSys2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.03275">Paper</a> | <a href="https://sandyresearch.github.io">Blogs</a> | <a href="https://www.youtube.com/watch?v=Rg9enIRSXmo">Video Tutorial</a></h3>
</p>

Diffusion transformers (DiTs) are bottlenecked by attention and MLP layers. What if we could make those layers faster? **Chipmunk is a training-free method to accelerate diffusion transformers with hardware-aware, training-free dynamic sparsity**. Chipmunk caches attention weights and MLP activations from previous steps and dynamically computes a sparse “delta” against the cached weights. We make Chipmunk hardware-efficient through [128, 1] and [192, 1] column-sparsity patterns \+ a suite of optimized sparse attention and MLP CUDA kernels.

_Developed in collaboration between Together AI, Hazy Research, and Sandy Research._

## 🎆 At a glance...

- **\~3.7x** faster video generation on 1xH100 HunyuanVideo at 720x1280 resolution for a 5s video (50 steps)
- **\~2.5x** faster video generation on 8xH100 HunyuanVideo at 720x1280 resolution for a 5s video (50 steps)
- **\~2.67x** faster video generation on 1xH100 Wan2.1 at 720x1280 resolution for a 3s video (50 steps)
- **\~1.6x** faster image generations on 1xH100 FLUX.1-dev at 1280x768 resolution (50 steps)
- Column Sparse Attention layer is **~9.3x** faster than FlashAttention3 baseline
- Column Sparse MLP layer is **~2.5x** faster than cuBLAS baseline

## 📸 Demos

https://github.com/user-attachments/assets/eb68abb6-249f-4e3a-96fe-657b7cf04531

<p align="center"><img src="assets/images/chipmunk-comparison.png" width="75%"></p>

<p align="center"><i>Images of cute chipmunks can be generated 1.37x faster! <b>Left</b>: Fully Dense FLUX.1-dev. <b>Right</b>: Ours (84% sparse attention and 70% sparse MLP)</i></p>

## ⏳ Update Log

- **6/15/2025:** We release a tutorial guide for adding Chipmunk to any DiT codebase! [Check it out here!](examples/YOUR-MODEL-HERE/README.md). Check out the video tutorial + method explanation on YouTube: https://www.youtube.com/watch?v=Rg9enIRSXmo.

- **6/14/2025:** Our attention kernels \[[1](csrc/attn/csp_attn.cu), [2](csrc/attn/dense_attn.cu), [3](csrc/attn/dense_colsum_attn.cu)\] now support completely unpadded and arbitrarily strided inputs for Q, K, and V. No more padding/`.contiguous()` calls necessary! This saves 5-10% of the E2E video generation latency.

- **6/13/2025:** We add official support for Mochi, achieving a 1.4x near-lossless speedup. [Check it out here!](examples/mochi)

- **6/11/2025:** Accepted to ES-FoMo III at ICML 2025.

- **6/09/2025:** Chipmunk's kernels are ported from CUDA to Triton, and we officially launch multi-architecture support! We test all models across Ampere and Hopper architectures, finding a comparable E2E generation speedup.

- **5/12/2025:** Presented at the YPS workshop at MLSys 2025.

## 💡 Quickstart

### 1\. Clone repo, build kernels, & install deps

```bash
git clone https://github.com/sandyresearch/chipmunk --recurse-submodules --shallow-submodules --depth 1

cd chipmunk
# Create a conda environment for the project
conda create -n chipmunk python=3.11 -y
conda activate chipmunk
conda install cuda==12.8.0 -c nvidia -y
# Install dependencies and build kernels
pip install -e . --no-build-isolation
```

Our kernels are written for Hopper GPUs, and depend on optimizations specific to CUDA Toolkit version ≥12.4 (we recommend 12.8\!).

### 2\. Make your GPUs go brr!

We currently support two models for acceleration, with a third coming soon. Keep in mind that for the first few image/video generations, it will be slower due to the cold start overhead of the PyTorch compiler. You should see speedups beginning at generation #3 and onwards.

#### 🎬 Hunyuan Video Generation Example

Use the one-line accelerated inference script to get started, and then check out [examples/hunyuan/README.md](examples/hunyuan/README.md) for a comprehensive tutorial.

```bash
cd examples/hunyuan
# Download weights
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava-llama-3-8b-v1_1-transformers
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ./ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ./ckpts/text_encoder
# One-line accelerated inference script
python3 sample_video.py --flow-reverse --chipmunk-config ./chipmunk-config.yml
```

For running on multiple H100s, see the [instructions](https://github.com/sandyresearch/chipmunk/blob/multigpu/Dockerfile) for building and running the Docker container on the multigpu branch.

_FYI: for Chipmunk's just-in-time offloading, we manage a pool of pinned CPU memory. Model initialization may take up to ~5 minutes as we allocate all these pinned buffers in RAM!_

#### 🎬 Wan2.1 Generation Example

Use the one-line accelerated inference script to get started, and then check out [examples/wan/README.md](examples/wan/README.md) for a comprehensive tutorial.

```bash
cd examples/wan
# Download weights
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
# One-line accelerated inference script
./run.sh
```

#### 🌅 FLUX.1-dev Image Generation Example

Use the one-line accelerated inference script to get started, and then check out [examples/flux/README.md](examples/flux/README.md) for a comprehensive tutorial.

```bash
cd examples/flux && pip install -e . && python -m flux.cli --name flux-dev --loop --prompt "A very cute cartoon chipmunk dressed up as a ninja holding katanas" --chipmunk-config ./chipmunk-config.yml
```

#### 🎥 Mochi Video Generation Example

Use the one-line accelerated inference script to get started, and then check out [examples/mochi/README.md](examples/mochi/README.md) for a comprehensive tutorial.

```bash
cd examples/mochi && python3 ./scripts/download_weights.py weights/
./run.sh
```

#### Want to add Chipmunk to another model that's not listed?

We've made a tutorial guide for you that will help you add Chipmunk to any DiT codebase! Check out [examples/YOUR-MODEL-HERE/README.md](examples/YOUR-MODEL-HERE/README.md) for a comprehensive tutorial. There's also a video version of this tutorial here:

<p align="center"><a href="https://www.youtube.com/watch?v=Rg9enIRSXmo"><img src="assets/images/yt-thumbnail.png" width="75%"></a></p>

## ⏰ Benchmarks

<p align="center"><img src="assets/images/speed.png" width="75%"></p>

Baselines: E2E models are `torch.compile`d from reference repositories. Attention layer uses FlashAttention3 as a backend. MLP layer uses torch compiled nn.Sequential (maximal performance with fused activations).

**Quality**
| **Method** | **Speedup ↑** | **Latency (s) ↓** | **Total ↑** | **Quality ↑** | **Semantic ↑** |
|------------|---------------|-------------------|-------------|---------------|----------------|
| ***`HunyuanVideo`, T = 50 (720×1280×129)*** | | | | | |
| Hunyuan                              | 1×  | 1030 | 83.24 | 85.09 | 75.82 |
| STA                                  | 1.79× | 575  | 82.46 | **84.63** | 73.83 |
| **Chipmunk**                         | **2.16×** | **477** | **82.94** | 84.60 | **76.3** |
| Step Caching (TeaCache)              | 3.69× | 279  | 80.79 | 82.87 | 72.5 |
| **Chipmunk + Step Cache** 1x H100    | **3.72×** | **277** | **82.5** | **84.23** | **75.6** |
| **Chipmunk + Step Cache** 8x H100    | **3.72×** | **277** | **82.5** | **84.23** | **75.6** |
| ***`WAN2.1`, T = 50 (720×1280×121)*** | | | | | |
| WAN2.1                               | 1×  | 1357 | 81.47 | 83.57 | 73.08 |
| STA                                  | 1.36× | 998  | **81.84** | **83.65** | **74.60** |
| **Chipmunk + STA**                   | **1.56×** | **870** | 81.71 | 83.61 | 74.12 |
| Step Caching (TeaCache)              | 2.0× | 678  | 81.17 | 83.24 | 72.87 |
| Chipmunk-56% + STA + Step Cache     | 2.20× | 616  | **81.73** | **83.74** | 73.69 |
| **Chipmunk-73% + STA + Step Cache** | **2.67×** | **508** | 81.11 | 82.88 | **74.05** |

*Performance comparison of various methods across different datasets for video generation.*  


| **Method** | **FLOPs ↓** | **Speedup ↑** | **Latency (s) ↓** | **ImRe ↑** |
|------------|-------------|---------------|-------------------|------------|
| ***`FLUX.1-dev`, T = 50 (768×1280)*** | | | | |
| Flux                           | 100% | 1×    | 6.60 | 0.76 |
| DiTFastAttn                    | 83%  | 1.09× | 6.05 | **0.80** |
| **Chipmunk**                   | **58%** | **1.41×** | **4.90** | **0.80** |
| Step + Token Caching (ToCa)    | 66%  | 1.51× | 4.37 | 0.76 |
| Step Caching (TeaCache)        | 39%  | 2.51× | 2.64 | 0.68 |
| **Chipmunk + Step Cache**      | **31%** | **2.56×** | **2.57** | **0.77** |

*Performance comparison of various methods on ImageReward (image generation).*

| **Method** | **FLOPs ↓** | **Speedup ↑** | **Latency (s) ↓** | **GenEval ↑** | **CLIP ↑** |
|------------|-------------|---------------|-------------------|---------------|------------|
| ***`FLUX.1-dev`, T = 50 (768×1280)*** | | | | | |
| Flux                           | 100% | 1×    | 6.60 | 0.66 | 31.07 |
| Step + Token Caching (ToCa)    | 66%  | 1.51× | 4.37 | 0.65 | 31.21 |
| Step Caching (TeaCache)        | 45%  | 2.23× | 2.95 | 0.61 | 31.37 |
| **Chipmunk-77% + Step Cache** | **31%** | **2.56×** | **2.57** | 0.62 | 31.18 |
| Chipmunk-65% + Step Cache     | 38%  | 2.25× | 2.93 | **0.66** | **31.43** |

*Performance comparison of various methods on GenEval and CLIP metrics.*  
*Note: Chipmunk-X% denotes a sparsity level of X% to assess the speed-quality trade-off.*

## 📖 How it Works

Chipmunk starts from two empirical facts about Diffusion Transformers: activations evolve slowly across timesteps, and both attention weights and MLP activations are highly sparse.

<p align="center"><img src="assets/images/howitworks-sum.png" width="60%"></p>
Leveraging this, it caches each layer's outputs from step n − 1 and, at step n, performs a "delta" pass that recomputes only the few vectors whose weights or values have materially changed, reusing the rest.   
<p align="center"><img src="assets/images/howitworks-cache.png" width="60%"></p>
Because GPUs excel at block‑sized work, Chipmunk maps these deltas onto block‑sparse patterns (e.g., 128× 256 tiles) that align with the hardware's GEMM kernels, skipping entire blocks instead of single elements. It then reorders keys, values, and tokens on the fly so that the sparse rows pack densely inside each tile, achieving an effective [128× 1] column sparsity while maintaining contiguous memory access.   
<p align="center"><img src="assets/images/howitworks-sram.png" width="60%"></p>

## 📚 Further Reading

### 🗒️ Technical Blog Posts

1. **[Overview](https://sandyresearch.github.io/chipmunk-part-I/)**: Overview of our sparsity method and what inspired it
2. **[Mathematical Theory](https://sandyresearch.github.io/chipmunk-part-II/)**: Builds mathematical intuition for the core ideas behind Chipmunk
3. **[GPU Optimization & Systems](https://sandyresearch.github.io/chipmunk-part-III/)**: A deep-dive on how Chipmunk exploits GPU kernel optimizations to become hardware-efficient

### 🙋‍♂️ Documentation

- **[Mochi Tutorial on YouTube](https://www.youtube.com/watch?v=Rg9enIRSXmo)**: See how Chipmunk is implemented into Mochi, and apply it to your favorite DiT model!
- **[Hunyuan Tutorial](examples/hunyuan/README.md)**: A tutorial of how to edit sparsity settings in Hunyuan and generate fast videos
- **[FLUX.1-dev Tutorial](examples/flux/README.md)**: A tutorial of how to edit sparsity settings in Flux and generate fast images
- **[Kernel Specification](csrc/README.md):** Description and purpose of each custom CUDA kernel if you'd like to start hacking on our kernels!
- **[Add Chipmunk to Your DiT Model](examples/YOUR-MODEL-HERE/README.md):** A written tutorial on how to add Chipmunk to any DiT codebase

<p align="center"><img src="assets/images/kittens.png" width="60%" /></p>

[howitworks-sum]: assets/images/howitworks-sum.png
[howitworks-cache]: assets/images/howitworks-cache.png
[howitworks-sram]: assets/images/howitworks-sram.png
[video-grid]: assets/videos/comparison-grid.mp4
[speed]: assets/images/speed.png
[comparison]: assets/images/chipmunk-comparison.png

## Citation

If you find this work useful, you can cite us as follows:

```
@misc{silveria2025chipmunktrainingfreeaccelerationdiffusion,
      title={Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas},
      author={Austin Silveria and Soham V. Govande and Daniel Y. Fu},
      year={2025},
      eprint={2506.03275},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.03275},
}
```
