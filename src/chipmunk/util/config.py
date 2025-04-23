import copy
import yaml

BASE_CONFIG = {
    'should_profile': False,
    'generation_index': 0,
    'steps': 50,
    # Multi-GPU currently only supported for Hunyuan
    'world_size': 1,

    'mlp': {
        'is_enabled': True,
        'is_fp8': False,

        'top_keys': "dd", # this will generate an error indicating that top_keys is not a float
        'random_keys': 0.05,
        'full_step_every': 10,
        'block_mask_cache': 2,
        'first_n_dense_layers': 2,

        # do not change below this line
        'counts_multiple_of': 256,
        'bm': 128,
        'mbm': 128,
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
        # If not None, will override full_step_every
        # 'full_step_schedule': set([0, 1, 10, 40]),
        'full_step_schedule': None,

        'recompute_mask': True,
        'should_compress_indices': True,
        
        # do not change below this line
        'counts_multiple_of': 128,
        'pad_qkv_before_kernel': True,
        'mbm': 192,
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
        'is_enabled': True,

        'skip_step_schedule': set([7, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43])
    }
}

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
