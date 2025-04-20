# 🐿️ Chipmunk: Adaptive Column-Sparse Deltas for 1.5-2x Faster Image & Video Gen

# Quickstart

## 1. Set up the repo

Our kernels rely on **H100**-specific features, so you will need to compile on a H100 GPU with at least CUDA Toolkit version 12.4 or greater (we recommend 12.8!).

```bash
# Clone w/ submodules! If you've already cloned it, run `git submodule update --init --recursive`
git clone --recurse-submodules --shallow-submodules --depth 1 https://github.com/sandyresearch/chipmunk chipmunk
cd chipmunk

# Create a conda environment for the project
conda create -n chipmunk python=3.11 -y
conda activate chipmunk
conda install cuda==12.8.0 -c nvidia -y # need cuda >12.4 for fast kernel performance!

# Install dependencies and build kernels
pip install -e .
```

## 2. Select a model

Currently, Chipmunk supports two models: **Hunyuan Video** and **FLUX.1-dev**. You can find these in the `models` directory.

### Hunyuan Video Generation Example
```bash
cd examples/hunyuan
python example_hunyuan.py
```

### FLUX.1-dev Image Generation Example
```bash
cd examples/flux
python example_flux.py
```

# ⚙️ Developing

- `csrc`: Includes kernels, including (1) attention kernels, (2) MLP kernels, and (3) indexed IO kernels (useful when dealing with sets of indices!). Our kernels are written in our favorite CUDA framework, [ThunderKittens](https://github.com/HazyResearch/ThunderKittens). Check them out! 🐈‍⬛
- `src`: Python bindings to kernels and plug-and-play PyTorch modules that incorporate them. See `src/chipmunk/modules/mlp.py` and `src/chipmunk/modules/attn.py` for the core building blocks of Chipmunk.
- `implementation`: Real implementations for models including Hunyuan Video, Flux, and Mochi. Use this to get a feel for how to incorporate CChipmunk into your own video generation pipeline!
