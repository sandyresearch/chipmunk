import copy
import yaml

BASE_CONFIG = {
    'num_model_invocations_per_inference_step': 1,
    'should_profile': False,
    'generation_index': 0,
    'steps': 50,
    # Multi-GPU currently only supported for Hunyuan
    'world_size': 1,

    'mlp': {
        'is_enabled': True,
        'is_fp8': False,

        'top_keys': 0.3,
        'random_keys': 0.05,
        'full_step_every': 10,
        'block_mask_cache': 2,
        'first_n_dense_layers': 2,

        'provider': 'cuda', # either 'cuda' or 'triton'
    },
     "patchify": {
        'is_enabled': True,

        # To disable patching at any level, set that level's patch size to 1. To disable patching entirely, set all patch sizes to 1.
        "chunk_size_1": 8,
        "chunk_size_2": 4,
    },
    'attn': {
        'is_enabled': True,
        'top_keys': 0.05,
        'random_keys': 0.01,
        'local_voxels': 0,
        'local_1d_window': 0,

        'first_n_dense_layers': 2,
        'full_step_every': 10,
        'full_step_schedule': None,

        'recompute_mask': True,
        'should_compress_indices': True,
        'should_keep_tail_dense': False,
        
        'provider': 'cuda', # either 'cuda' or 'triton'
    },
    "offloading": {
        'global_disable_offloading': False,

        'mlp.out_cache': False,
        'mlp.indices': False,
        'mlp.counts': False,
        'mlp.sparse_act_T': False,
        'mlp.blockmean_mid_cache': False,

        'attn.out_cache': True,
        'attn.indices': True,
        'attn.counts': False,
        'attn.lse_constants': False,

        'text_encoders': True,
    },
    "step_caching": {
        'is_enabled': False,
        'skip_step_schedule': set([7, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43])
    }
}

def get_kernel_config_mlp():
    # use the same block sizes for MLP across triton and CUDA implementations
    return {
        'bm': 128,
        'mbm': 128,
        'counts_multiple_of': 256,
    }

def get_kernel_config_attn():
    if GLOBAL_CONFIG['attn']['provider'] == 'triton': 
        # Triton-based FA2 uses a blocksize of 64x64 but we still use 192 for the BM for memory efficiency
        return {
            'bm': 192,
            'counts_multiple_of': 64,
            'indices_pad_to': 1,
        }
    elif GLOBAL_CONFIG['attn']['provider'] == 'cuda':
        # CUDA-based FA3 uses a blocksize of 192x~128
        return {
            'bm': 192,
            'counts_multiple_of': 112,
            'indices_pad_to': 4,
        }
    else:
        raise ValueError(f"Invalid provider: {GLOBAL_CONFIG['attn']['provider']}")

GLOBAL_CONFIG = copy.deepcopy(BASE_CONFIG)

def update_global_config(config):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG.update({
        **GLOBAL_CONFIG,
        **config,
    })

import sys
import yaml
from typing import Dict, Any

def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> None:
    """Recursively update dictionary d with values from u"""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v

def load_from_file(config_file: str) -> None:
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    # Update global config
    if yaml_config:
        _deep_update(GLOBAL_CONFIG, yaml_config)
        # update_global_config(yaml_config)
        print(f"CHIPMUNK: using config file {config_file}")