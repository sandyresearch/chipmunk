GLOBAL_CONFIG = {
    'mlp': {
        'is_enabled': True,
        'top_keys': 0.3,
        'random_keys': 0.05,
        'full_step_every': 10,
        'block_mask_cache': 1,
        'first_n_dense_layers': 2,

        # do not change below this line
        'counts_multiple_of': 256,
        'bm': 128,
        'mbm': 16,
    },
     "patchify": {
        # To disable patching at any level, set that level's patch size to 1. To disable patching entirely, set all patch sizes to 1.
        "chunk_size_1": 8,
        "chunk_size_2": 4,
    },
    'attn': {
        'is_enabled': True,
        'top_keys': 0.165,
        'full_step_every': 10,
        'first_n_dense_layers': 2,

        # do not change below this line
        'counts_multiple_of': 112, # the # of kv_tile_rows in csrc/attn/csp_attn.cu
        'mbm': 192,
    },
    "offloading": {
        'global_disable_offloading': False,

        'mlp.out_cache': False,
        'mlp.indices': False,
        'mlp.counts': False,
        'mlp.sparse_act_T': False,
        'mlp.blockmean_mid_cache': False,

        'attn.out_cache': False,
        'attn.indices': False,
        'attn.counts': False,
        'attn.lse_constants': False,
    },

}
