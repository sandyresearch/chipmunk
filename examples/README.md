# Chipmunk-Enabled Models

We've implemented Chipmunk into the following models:

- FLUX.1-dev
- HunyuanVideo
- WAN2.1
- Mochi

## Add your model

See [YOUR-MODEL-HERE/README.md](YOUR-MODEL-HERE/README.md) for a tutorial! We welcome open-source contributions and PRs.

## Features

| Model          | Sparse Attention | Sparse MLP | Token Reordering |
|----------------|------------------|------------|------------------|
| FLUX.1-dev     | ✅               | ✅         | ✅               |
| HunyuanVideo   | ✅               | ❌         | ✅               |
| WAN2.1         | ✅               | ❌         | ✅               |
| Mochi          | ✅               | ❌         | ✅               |

_Note: Most video models spend only ~10% of their total runtime on MLP as opposed to 70%+ on attention, so we only sparsify attention on video models!_
