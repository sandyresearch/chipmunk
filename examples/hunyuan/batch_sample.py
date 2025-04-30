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

from chipmunk.util.config import GLOBAL_CONFIG

chipmunk_world_size = int(os.environ.get('CHIPMUNK_WORLD_SIZE', 1))
chipmunk_local_rank = int(os.environ.get('CHIPMUNK_LOCAL_RANK', 0))

@ray.remote(num_gpus=1)
def main(args=None, local_rank=None, world_size=None):
    chipmunk.util.config.load_from_file(args.chipmunk_config)
    args.flow_reverse = True
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise ValueError(f"`prompt_file` not exists: {prompt_file}")
    prompts = json.load(open(prompt_file, 'r'))
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    # save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    save_path = 'outputs/chipmunk-test/'
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
    hunyuan_video_sampler.pipeline.transformer.sparsify()

    # Get the updated args
    args = hunyuan_video_sampler.args

    for i, generation in enumerate(prompts):
        if i % chipmunk_world_size != int(chipmunk_local_rank):
            continue
        prompt_text = generation['prompt']
        seed = random.randint(0, 1000000)
        has_valid_output_path = False
        for output_path in generation['output_path']:
            cur_save_path = Path(args.chipmunk_config).parent / 'media' / output_path
            if not cur_save_path.exists():
                has_valid_output_path = True
                break
        if not has_valid_output_path: # skip if all output paths have been generated already
            continue

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
        logger.info(f'finished in {outputs["gen_time"]:.3f}s')
        
        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                for output_path in generation['output_path']:
                    cur_save_path = Path(args.chipmunk_config).parent / 'media' / output_path
                    Path(cur_save_path).parent.mkdir(parents=True, exist_ok=True)
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
    ray.init(_temp_dir='/tmp/ray-hunyuan')
    args = parse_args()
    chipmunk.util.config.load_from_file(args.chipmunk_config)
    args.ulysses_degree = GLOBAL_CONFIG['world_size']
    args.flow_reverse = True
    results = run_all(args)

