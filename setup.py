import os
import subprocess
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

HOPPER_GENERATION = 90
DEFAULT_GENERATION = -1

sources = {
    # These kernels are only available on Hopper. There are equivalent Triton implementations for non-Hopper generations.
    'colsum_attn': {
        'source_files': { HOPPER_GENERATION: 'csrc/attn/dense_colsum_attn.cu' }
    },
    'csp_attn': {
        'source_files': { HOPPER_GENERATION: 'csrc/attn/csp_attn.cu' }
    },
    'attn': {
        'source_files': { HOPPER_GENERATION: 'csrc/attn/dense_attn.cu' }
    },
    'csp_mlp_mm1': {
        'source_files': { HOPPER_GENERATION: 'csrc/mlp/csp_mlp_mm1.cu' }
    },
    'csp_mlp_mm2_and_scatter_add': {
        'source_files': { HOPPER_GENERATION: 'csrc/mlp/csp_mlp_mm2_and_scatter_add.cu' }
    },
    'csp_scatter_add': {
        'source_files': { HOPPER_GENERATION: 'csrc/indexed_io/scatter_add.cu' }
    },
    # These kernels are available on all generations! Not just Hopper.
    'copy_indices': {
        'source_files': { DEFAULT_GENERATION: 'csrc/indexed_io/copy_indices.cu' }
    },
    'topk_indices': {
        'source_files': { DEFAULT_GENERATION: 'csrc/indexed_io/topk_indices.cu' }
    },
    'mask_to_indices': {
        'source_files': { DEFAULT_GENERATION: 'csrc/indexed_io/mask_to_indices.cu' }
    },
}

kernels = [
    'colsum_attn',
    'csp_attn',
    'attn',
    'csp_mlp_mm1',
    'csp_mlp_mm2_and_scatter_add',
    'csp_scatter_add',
    'copy_indices',
    'topk_indices',
    'mask_to_indices',
]

target = HOPPER_GENERATION if torch.cuda.get_device_capability()[0] == 9 else DEFAULT_GENERATION

tk_root = 'submodules/ThunderKittens'
tk_root = os.path.abspath(tk_root)
if not os.path.exists(tk_root):
    raise FileNotFoundError(f'ThunderKittens root directory {tk_root} not found.')
tk_include = f'{tk_root}/include'
if not os.path.exists(tk_include):
    raise FileNotFoundError(f'ThunderKittens include directory {tk_include} not found - please be sure to install all submodules to this folder.')

python_include = subprocess.check_output([
    'python', '-c', "import sysconfig; print(sysconfig.get_path('include'))"
]).decode().strip()
torch_include = subprocess.check_output([
    'python', '-c',
    "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"
]).decode().strip()
# CUDA flags
cuda_flags = [
    '-DNDEBUG', '-Xcompiler=-Wno-psabi', '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda', '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler', '--use_fast_math', '-std=c++20',
    '-O3', '-Xnvlink=--verbose', '-Xptxas=--verbose', '-lineinfo',
    '-Xptxas=--warn-on-spills',
    '-DTORCH_COMPILE',
] + torch_include.split()
cpp_flags = ['-std=c++20', '-O3', '-DDPy_LIMITED_API=0x03100000']

if target == HOPPER_GENERATION:
    cuda_flags.append('-DKITTENS_HOPPER')
    cpp_flags.append('-DKITTENS_HOPPER')

arch = f'sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}'
if arch == 'sm_90': arch = 'sm_90a'
cuda_flags.append(f'-arch={arch}')

source_files = ['csrc/chipmunk.cu']

for k in kernels:
    src_files = sources[k]['source_files']
    if target not in src_files and DEFAULT_GENERATION not in src_files:
        print(f'Warning: Target {target} not found in source files for kernel {k}. We will fallback to a Triton-based implementation.')
        continue
    if target in src_files:                       # exact match, e.g. Hopper on Hopper
        source_files.append(src_files[target])
    elif DEFAULT_GENERATION in src_files:         # portable implementation exists
        source_files.append(src_files[DEFAULT_GENERATION])
    else:                                         # neither variant exists → skip
        raise ValueError(f'No CUDA source for kernel {k} on target {target}')
    cpp_flags.append(f'-DTK_COMPILE_{k.replace(" ", "_").upper()}')

setup(
    name='chipmunk',
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            'chipmunk.cuda',
            sources=source_files,
            extra_compile_args={
                'cxx': cpp_flags,
                'nvcc': cuda_flags
            },
            include_dirs=[
                python_include,
                torch_include,
                f'{tk_root}/include',
                f'{tk_root}/prototype',
            ],
            libraries=['cuda', 'cublas', 'cudart', 'cudadevrt'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}}      
)
