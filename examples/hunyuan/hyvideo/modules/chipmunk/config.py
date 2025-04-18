import copy

BASE_CONFIG = {
    'head_parallel': True,
    'voxel_order': True,
}

GLOBAL_CONFIG = copy.deepcopy(BASE_CONFIG)

def update_global_config(config):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG.update({
        **BASE_CONFIG,
        **config,
    })
