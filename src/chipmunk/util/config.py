GLOBAL_CONFIG = {
    'mlp': {
        'top_keys': 0.3,
        'random_keys': 0.05,
        'full_step_every': 10,
        'block_mask_cache': 1,

        # do not change below this line
        'counts_multiple_of': 256,
        'mbm': 128,
        'bm': 128,
    },
    'attn': {
        'top_keys': 0.165,
        'full_step_every': 10,

        # do not change below this line
        'counts_multiple_of': 192,
        'mbm': 192,
    },
    "offloading": {
        'global_disable_offloading': False,

        'mlp.out_cache': True,
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
