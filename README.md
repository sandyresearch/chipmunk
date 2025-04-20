# Chipmunk: Hardware-Aware Sparsity for Accelerated Video & Image Generation

Diffusion transformers (DiTs) are bottlenecked by attention and MLP layers. What if we could make those layers faster? **Chipmunk is a training-free method to accelerate diffusion transformers with hardware-aware, training-free dynamic sparsity**. Chipmunk caches attention weights and MLP activations from previous steps and dynamically computes a sparse “delta” against the cached weights. We make Chipmunk hardware-efficient through \[128, 1\] and \[192, 1\] column-sparsity patterns \+ a suite of optimized attention and MLP kernels that are up to 9.3x and 2.5x faster than FlashAttention3 and cuBLAS, respectively. 

⌚️Stats:

- **\~3.7x** faster video generation on HunyuanVideo at 720x1280 resolution for a 5s video (50 steps)  
- **\~1.4x** faster image generations on FLUX.1-dev at 1280x768 resolution (50 steps)  
- Column Sparse Attention layer is **9.3x** faster than FlashAttention3 baseline  
- Column Sparse MLP layer is **2.5x** faster than cuBLAS baseline

## Demos


![][comparison]
<p align="center"><i>Images of cute chipmunks can be generated 1.37x faster! Left: Fully Dense FLUX.1-dev. Right: Ours (84% sparse attention and 70% sparse MLP)</i></p>


## Quickstart

### 1\. Clone repo, build kernels, & install deps

```
git clone https://github.com/sandyresearch/chipmunk --recurse-submodules --shallow-submodules --depth 1

cd chipmunk

# Create a conda environment for the project
conda create -n chipmunk python=3.11 -y
conda activate chipmunk
conda install cuda==12.8.0 -c nvidia -y # need cuda >12.4 for fast kernel performance!

# Install dependencies and build kernels
pip install -e .
pip install -e ./examples/hunyuan # Hunyuan video generation
pip install -e ./examples/flux # Flux video generation
```

Our kernels are written for Hopper GPUs, and depend on optimizations specific to CUDA Toolkit version ≥12.4 (we recommend 12.8\!).

### 2\. Select a model

Currently, Chipmunk supports two models: **Hunyuan Video** and **FLUX.1-dev**. You can find these in the `models` directory.

#### Hunyuan Video Generation Example

Use the **one-line accelerated inference script** to get started, and then check out examples/hunyuan/README.md for a comprehensive tutorial.

```
cd examples/hunyuan && python -m <example script>
```

#### FLUX.1-dev Image Generation Example

Use the **one-line accelerated inference script** to get started, and then check out examples/flux/README.md for a comprehensive tutorial.

```
cd examples/flux && python -m flux.cli --name flux-dev --prompt "A very cute cartoon chipmunk dressed up as a ninja holding katanas"
```

## How it Works

Chipmunk starts from two empirical facts about Diffusion Transformers: activations evolve slowly across timesteps, and both attention weights and MLP activations are highly sparse.   
![][howitworks-sum]
Leveraging this, it caches each layer’s outputs from step n − 1 and, at step n, performs a “delta” pass that recomputes only the few vectors whose weights or values have materially changed, reusing the rest.   
![][howitworks-cache]  
Because GPUs excel at block‑sized work, Chipmunk maps these deltas onto block‑sparse patterns (e.g., 128 × 256 tiles) that align with the hardware’s GEMM kernels, skipping entire blocks instead of single elements. It then reorders keys, values, and tokens on the fly so that the sparse rows pack densely inside each tile, achieving an effective \[128 × 1\] column sparsity while maintaining contiguous memory access.   
![][howitworks-sram]

## Benchmarks

![][image4]

**Quality**

| Hunyuan | VBench Quality | VB Semantic | VB Total | Resolution | Sparsity | Latency | Speedup |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| FlashAttention-3 | 85.09% | 75.82% | 83.24% | 720x1280x129 | 0% | 1030s | 1x |
| Sliding Tile Attention (Training-Free) | 84.63% | 73.83% | 82.46% | 768x1280x117 | 58% | 945s \-\> 527s | 1.79x |
| Chipmunk (Training-Free) | 84.60% | 76.29% | 82.94% | 720x1280x129 | 82% \* | 1030s \-\> 477s | 2.16x |
| Chipmunk \+ Step Caching (Training-Free) | 84.22% | 75.60% | 82.50% | 720x1280x129 | 87% | 1030s \-\> 277s | 3.72x |

 \* 93% sparsity on 44 out of 50 steps for an average of 82% sparsity.

| FLUX.1-dev\* (bf16) | ImageReward | MLP Sparsity | Attn Sparsity | Speedup |
| :---- | :---- | :---- | :---- | :---- |
| Baseline (with FlashAttention-3) | 76.6% | 0% | 0% | 1x |
| Chipmunk | 80.2%	 | 70% | 83.5% | **1.37x** |
| Chipmunk \+ Step Caching | 78.0% | 70% | 83.5% | **1.63x** |

## Further Reading

Technical Deep Dives:

1. Summary: See SandyResearch blog  
2. Theory: See SandyResearch blog  
3. Systems: See SandyResearch blog

Tutorials

* **Kernel Specification (csrc/README.md):** Description and purpose of each custom CUDA kernel if you’d like to start hacking on our kernels\!  
* **Hunyuan Video Tutorial (examples/hunyuan/README.md)**: A tutorial of how to edit sparsity settings in Hunyuan and generate fast videos  
* **FLUX.1-dev Tutorial (examples/flux/README.md)**: A tutorial of how to edit sparsity settings in Flux and generate fast images

## Contributors

Austin Silveria, Soham Govande, Dan Fu

[howitworks-sum]: assets/images/howitworks-sum.png
[howitworks-cache]: assets/images/howitworks-cache.png
[howitworks-sram]: assets/images/howitworks-sram.png
[video-grid]: assets/videos/comparison-grid.mp4
[comparison]: assets/images/chipmunk-comparison.png