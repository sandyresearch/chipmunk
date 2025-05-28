import os
import subprocess
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

sources = {
    'colsum_attn': {
        'source_files': {
            'h100': 'csrc/attn/dense_colsum_attn.cu'
        }
    },
    'csp_attn': {
        'source_files': {
            'h100': 'csrc/attn/csp_attn.cu'
        }
    },
    'csp_128_attn': {
        'source_files': {
            'h100': 'csrc/attn/csp_128_attn.cu'
        }
    },
    'attn': {
        'source_files': {
            'h100': 'csrc/attn/dense_attn.cu'
        }
    },
    'csp_mlp_mm1': {
        'source_files': {
            'h100': 'csrc/mlp/csp_mlp_mm1.cu'
        }
    },
    'csp_scatter_add': {
        'source_files': {
            'h100': 'csrc/indexed_io/scatter_add.cu'
        }
    },
    'copy_indices': {
        'source_files': {
            'h100': 'csrc/indexed_io/copy_indices.cu'
        }
    },
    'topk_indices': {
        'source_files': {
            'h100': 'csrc/indexed_io/topk_indices.cu'
        }
    },
    'mask_to_indices': {
        'source_files': {
            'h100': 'csrc/indexed_io/mask_to_indices.cu'
        }
    },
    'csp_mlp_mm2_and_scatter_add': {
        'source_files': {
            'h100': 'csrc/mlp/csp_mlp_mm2_and_scatter_add.cu'
        }
    }
}

kernels = [
    'colsum_attn',
    'csp_attn',
    'csp_128_attn',
    'attn',
    'csp_mlp_mm1',
    'csp_mlp_mm2_and_scatter_add',
    'csp_scatter_add',
    'copy_indices',
    'topk_indices',
    'mask_to_indices',
]

target = 'h100'

tk_root = 'submodules/ThunderKittens'
tk_root = os.path.abspath(tk_root)
if not os.path.exists(tk_root):
    raise FileNotFoundError(f'ThunderKittens root directory {tk_root} not found - please be sure to install all submodules to this folder.')

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
cpp_flags = ['-std=c++20', '-O3', '-DDPy_LIMITED_API=0x03110000']

if target == 'h100':
    cuda_flags.append('-DKITTENS_HOPPER')
    cuda_flags.append('-arch=sm_90a')
else:
    raise ValueError(f'Target {target} not supported')

source_files = ['csrc/chipmunk.cpp']

for k in kernels:
    if target not in sources[k]['source_files']:
        raise KeyError(f'Target {target} not found in source files for kernel {k}')
    if isinstance(sources[k]['source_files'][target], list):
        source_files.extend(sources[k]['source_files'][target])
    else:
        source_files.append(sources[k]['source_files'][target])
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
    options={"bdist_wheel": {"py_limited_api": "cp311"}}      
)