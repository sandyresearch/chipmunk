

```bash
# Clone it with submodules! If you've already cloned it, run `git submodule update --init --recursive`
git clone --recurse-submodules --shallow-submodules --depth 1 https://github.com/sandyresearch/chipmunk chipmunk
cd chipmunk
# Create a conda environment for the project
conda create -n chipmunk python=3.11 -y
conda activate chipmunk
conda install cuda==12.8.0 -c nvidia -y # need cuda >12.4 for fast kernel performance!
# Install all dependencies
pip install .
```
