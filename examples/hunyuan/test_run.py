import os
import json
import subprocess
import torch

HF_HOME = '/data/austin/hf_cache'

# BASE_CONFIG = {
#     'voxel_order': True,
#     'head_parallel': True,
#     'tk': 1,
#     'lk': 0,
#     'rk': 0,
#     'pli': 1,
#     'steps': 50,

#     'savedir': 'chipmunk',
# }

BASE_CONFIG = {
    "head_parallel": False,
    "world_size": 1,
    "steps": 4,
    "offload": 1,

    "voxel_order": False,
    "tk": 1,
    "lk": 0,
    "rk": 0,

    "savedir": "dense",

    # TODO: delete
    "pli": 1,
    "eval": "custom-one-seed",
    "tk-dense": 1,
  }

BASE_CONFIG = {
    "steps": 4,
    "world_size": 1,
    "head_parallel": True,
    "compile": True,
    "offload": True,

    "voxel_order": True,
    "tk": 1,
    "lk": 0,
    "rk": 0.01,
    "start_step": 2,
    "start_layer": 0,
    "full_every": 10,
    "lv": 0,
    "topk": 0.05,

    "savedir": "chipmunk",

    # TODO: delete
    "eval": "custom-one-seed",
    "pli": 1,
    "tk-sparse-diff": 1,
}

CONFIG_DIR_PREFIX = "./outputs"

def run(configs):
    for config in configs:
        config_dir = f"{CONFIG_DIR_PREFIX}/{config['savedir']}"
        os.makedirs(config_dir, exist_ok=True)
        config_path = f'{config_dir}/config.json'
        with open(config_path, 'w') as f:
            json.dump([config], f, indent=2)
        log_path = f'{config_dir}/logs'
        os.makedirs(log_path, exist_ok=True)

        print(f"Starting generation process")
        env = os.environ.copy()
        env.update({
            'HF_HOME': HF_HOME
        })
        
        world_size = config['world_size'] if 'world_size' in config else 1
        with open(f'{log_path}/gen.log', 'w') as log_file:
            cmd = [
                'python3',
                'sample_video.py',
                '--video-size', '720', '1280',
                '--video-length', '129',
                '--infer-steps', str(config['steps']),
                '--prompt', 'A cat walks on the grass, realistic style.',
	            '--seed', '42',
                '--embedded-cfg-scale', '6.0',
                '--flow-shift', '7.0',
                '--flow-reverse',
                '--ulysses-degree', str(world_size),
                '--ring-degree', '1',
                '--save-path', config_dir,
                '--config-path', config_path,
            ]

            p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        print(f"Generation process started")

        p.wait()

# load expected tensor
expected_tensor = torch.load('hunyuan_expected.pt')

run([BASE_CONFIG])

# read actual tensor
actual_tensor = torch.load('hunyuan_actual.pt')

# print max diff, average diff
print(f"Max diff: {torch.max(torch.abs(expected_tensor - actual_tensor))}")
print(f"Average diff: {torch.mean(torch.abs(expected_tensor - actual_tensor))}")

# assert all close
assert torch.allclose(expected_tensor, actual_tensor)
