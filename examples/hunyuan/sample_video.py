import chipmunk.util.config

import os
import json
import time
import random
from pathlib import Path
from loguru import logger
from datetime import datetime
import ray
import sys
import torch
import torch.distributed as dist
import torch._dynamo

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.chipmunk.config import update_global_config

from hyvideo.modules.head_parallel import setup_dist

# @ray.remote(num_gpus=1)
def main(args=None, local_rank=None, world_size=None):
    chipmunk.util.config.load_from_file("chipmunk-config.yml")
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    # save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    save_path = 'outputs/chipmunk'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # ==================== Initialize Distributed Environment ================
    device = torch.device(f"cuda")
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ['LOCAL_RANK'] = str(local_rank)
        # Ray initializes each process to only have one visible GPU.
        dist.init_process_group(
            "nccl",
            rank=local_rank,
            world_size=world_size,
        )
        pg = dist.group.WORLD
        setup_dist(pg, local_rank, world_size)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args, device=device)

    # Get the updated args
    args = hunyuan_video_sampler.args

    # for prompt in prompts:
    # twice for torch compile warmup
    # for prompt in [args.prompt, args.prompt]:
    prompt_cache = None
    while True:
        if prompt_cache is None:
            prompt = input("Enter a prompt: ")
            seed = input("Enter a seed (empty for random): ")
            if seed == "":
                seed = random.randint(0, 1000000)
            else:
                seed = int(seed)
            prompt_cache = prompt
        else:
            prompt = input("Enter a prompt (empty for previous prompt): ")
            if prompt == "":
                prompt = prompt_cache
            prompt_cache = prompt
            seed = input("Enter a seed (empty for random): ")
            if seed == "":
                seed = random.randint(0, 1000000)
            else:
                seed = int(seed)
        # prompt_ids = prompt['ids']
        # prompt_text = prompt['prompt']
        # seed = prompt['seed']

        # prompt_ids = [f"{args.prompt[:100].replace('/','')}-{args.seed:04d}"]
        # prompt_text = args.prompt
        # seed = args.seed
        prompt_ids = [f"{prompt[:100].replace('/','')}-{seed:04d}"]
        prompt_text = prompt
        seed = seed

        # Start sampling
        # TODO: batch inference check
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt_text, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        samples = outputs['samples']
        
        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                for prompt_id in prompt_ids:
                    cur_save_path = f"{save_path}/{prompt_id}.mp4"
                    save_videos_grid(sample, cur_save_path, fps=24)
                    logger.info(f'Sample save to: {cur_save_path}')

def run_all(args):
    import traceback
    import sys

    # Create a list of all actors (tasks) you’ll run in parallel
    actors = [main.remote(args=args, local_rank=gpu_id, world_size=args.ulysses_degree)
              for gpu_id in range(args.ulysses_degree)]

    try:
        # If one of these dies, ray.get will raise an exception
        results = ray.get(actors)
        return results
    except Exception as e:
        # Log what happened
        traceback.print_exc()

        # Kill all actors to avoid “zombie” processes
        for a in actors:
            ray.kill(a)

        # Optionally shut down Ray altogether
        ray.shutdown()

        # Now exit so that everything stops
        sys.exit(1)

if __name__ == "__main__":
    # ray.init(_temp_dir='/tmp/ray-hunyuan')
    args = parse_args()
    main(args, local_rank=0, world_size=1)
    # results = run_all(args)

