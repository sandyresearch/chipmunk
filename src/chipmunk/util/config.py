GLOBAL_CONFIG = {
    'mlp': {
        'top_keys': 0.3,
        'random_keys': 0.05,
        'full_step_every': 10,
        'block_mask_cache': 1,

        # do not change below this line
        'mbm': 128,
        'bm': 128,
    },
    'attn': {
        'top_keys': 0.165,
        'full_step_every': 10,

        # do not change below this line
        'mbm': 192,
    },
}
