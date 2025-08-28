import copy

BASE_CONFIG = {
    'head_parallel': True,
    'voxel_order': True,
}

HUNYUAN_GLOBAL_CONFIG = copy.deepcopy(BASE_CONFIG)

def update_global_config(config):
    global HUNYUAN_GLOBAL_CONFIG
    HUNYUAN_GLOBAL_CONFIG.update({
        **BASE_CONFIG,
        **config,
    })
