from __future__ import annotations

"""Chipmunk batch‑sampling harness with live tqdm progress bars.

The script iterates over a grid of Chipmunk configs, launches one
`batch_sample.py` process per GPU for each config, and shows a progress bar
that refreshes once per second while counting the files already produced in
`<exp_dir>/media/<eval_name>` against the number of prompts.

Execution moves on to the next config when either of these is true:
1. The number of media files equals the prompt count (bar hits 100 %).
2. All spawned child processes have terminated (even if they failed early).

If any child exits with a non‑zero code, the harness stops with that code.
"""

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm.auto import tqdm

from chipmunk.util.config import GLOBAL_CONFIG, load_from_file

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]  # repository root
EVAL_ROOT = ROOT_DIR / "evals"
PROMPT_DIR = EVAL_ROOT / "prompts"
MODEL_DIR = EVAL_ROOT / "models"

# Directory names in `examples/` may not exactly match the `--model` flag.
# Map any mismatches here.
EXAMPLE_DIR_MAP = {
    "hidream": "hidream-l1",
}

# -----------------------------------------------------------------------------
# Config generation (must be implemented by the user for new models)
# -----------------------------------------------------------------------------


def make_config(
    base_path: str,
    patchify: bool,
    attn_sparsity: float,
    attn_full_step_every: int,
    attn_full_step_schedule: set[int],
    attn_local_voxels: int,
    attn_local_1d_window: int,
    attn_recompute_mask: bool,
    mlp_sparsity: float,
    mlp_rk: float,
    mlp_mbm: int,
    mlp_is_fp8: bool,
    mlp_full_step_every: int,
    mlp_block_mask_cache: int,
    step_caching: bool,
    skip_step_schedule: set[int],
    width: int,
    height: int,
    global_disable_offloading: bool,
    tea_cache_threshold: float = 0.0,
    attn_rk: float = 0.01,
    world_size: int = 1,
):
    """Return a **list** with a single deep‑copied GLOBAL_CONFIG variant."""

    load_from_file(base_path)

    GLOBAL_CONFIG["patchify"]["is_enabled"] = patchify
    GLOBAL_CONFIG["attn"]["top_keys"] = attn_sparsity
    GLOBAL_CONFIG["attn"]["is_enabled"] = float(attn_sparsity) != 0.0 or float(attn_local_1d_window) != 0.0 or float(attn_local_voxels) != 0.0
    GLOBAL_CONFIG["attn"]["full_step_every"] = attn_full_step_every
    GLOBAL_CONFIG["attn"]["full_step_schedule"] = (
        attn_full_step_schedule if attn_full_step_schedule else None
    )
    GLOBAL_CONFIG["attn"]["local_voxels"] = attn_local_voxels
    GLOBAL_CONFIG["attn"]["local_1d_window"] = attn_local_1d_window
    GLOBAL_CONFIG["attn"]["recompute_mask"] = attn_recompute_mask

    GLOBAL_CONFIG["mlp"]["is_enabled"] = float(mlp_sparsity) != 0.0
    GLOBAL_CONFIG["mlp"]["top_keys"] = mlp_sparsity
    GLOBAL_CONFIG["mlp"]["random_keys"] = mlp_rk
    GLOBAL_CONFIG["mlp"]["mbm"] = mlp_mbm
    GLOBAL_CONFIG["mlp"]["is_fp8"] = mlp_is_fp8
    GLOBAL_CONFIG["mlp"]["full_step_every"] = mlp_full_step_every
    GLOBAL_CONFIG["mlp"]["block_mask_cache"] = mlp_block_mask_cache

    GLOBAL_CONFIG["step_caching"]["is_enabled"] = step_caching
    GLOBAL_CONFIG["step_caching"]["skip_step_schedule"] = skip_step_schedule

    GLOBAL_CONFIG["width"] = width
    GLOBAL_CONFIG["height"] = height

    GLOBAL_CONFIG["offloading"]["global_disable_offloading"] = global_disable_offloading

    if tea_cache_threshold > 0.0:
        GLOBAL_CONFIG["tea_cache"]["threshold"] = tea_cache_threshold
        GLOBAL_CONFIG["tea_cache"]["is_enabled"] = True
        GLOBAL_CONFIG["tea_cache"]["debug"] = True

    GLOBAL_CONFIG["attn"]["random_keys"] = attn_rk
    
    GLOBAL_CONFIG["world_size"] = world_size

    from copy import deepcopy

    return [deepcopy(GLOBAL_CONFIG)]


# -----------------------------------------------------------------------------
# Example grid for flux (expand / modify for other models)
# -----------------------------------------------------------------------------

def generate_configs_flux() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    for attn_sparsity in [0.1, 0.165, 0.3]:
        for mlp_sparsity in [0.0, 0.3]:
            for mlp_rk in [0.05]:
                for mlp_mbm in [16, 128]:
                    for recompute_mask in [False, True]:
                        for mlp_is_fp8 in [True, False]:
                            if mlp_is_fp8 != (mlp_sparsity == 0.0): continue
                            if mlp_is_fp8 and not recompute_mask: continue
                            
                            for w, h in [(1280, 768), (1536, 1536)]:
                                # only large image sizes on fp8
                                if w == 1280 != (not mlp_is_fp8): continue

                                for attn_full_step_every in [10, 20]:
                                    for mlp_full_step_every in [10, 20]:
                                        if mlp_full_step_every > 10 and mlp_is_fp8: continue
                                        for mlp_block_mask_cache in [1, 2, 3]:
                                            if mlp_block_mask_cache > 1 and mlp_is_fp8: continue
                                            for step_caching in [False, True]:
                                                cfgs += make_config(
                                                    base_path="examples/flux/chipmunk-config.yml",
                                                    patchify=True,
                                                    attn_sparsity=attn_sparsity,
                                                    attn_full_step_every=attn_full_step_every,
                                                    attn_recompute_mask=recompute_mask,
                                                    mlp_sparsity=mlp_sparsity,
                                                    mlp_rk=mlp_rk,
                                                    mlp_mbm=mlp_mbm,
                                                    mlp_is_fp8=mlp_is_fp8,
                                                    mlp_full_step_every=mlp_full_step_every,
                                                    mlp_block_mask_cache=mlp_block_mask_cache,
                                                    step_caching=step_caching,
                                                    skip_step_schedule={7, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43},
                                                    width=w,
                                                    height=h,
                                                    global_disable_offloading=True,
                                                    attn_full_step_schedule={},
                                                    attn_local_voxels=0,
                                                    attn_local_1d_window=0,
                                                )
    return cfgs

def generate_configs_hunyuan() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    cfgs += make_config(
        base_path="examples/hunyuan/chipmunk-config.yml",
        patchify=False,
        attn_sparsity=0.0,
        attn_full_step_every=1,
        attn_full_step_schedule={0, 1, 10, 40},
        attn_recompute_mask=True,
        mlp_sparsity=0,
        mlp_rk=0,
        mlp_mbm=0,
        mlp_is_fp8=False,
        mlp_full_step_every=1,
        mlp_block_mask_cache=0,
        step_caching=True,
        skip_step_schedule={7, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43},
        width=1280,
        height=720,
        global_disable_offloading=False,
        attn_local_voxels=0,
        attn_local_1d_window=0.1,
        world_size=1,
        attn_rk=0.01
    )
    # Tea Cache Config
    cfgs += make_config(
        base_path="examples/hunyuan/chipmunk-config.yml",
        patchify=False,
        attn_sparsity=0.0,
        attn_full_step_every=1,
        attn_recompute_mask=False,
        mlp_sparsity=0,
        mlp_rk=0,
        mlp_mbm=0,
        mlp_is_fp8=False,
        mlp_full_step_every=1,
        mlp_block_mask_cache=0,
        step_caching=False,
        skip_step_schedule={},
        width=1280,
        height=720,
        global_disable_offloading=True,
        attn_full_step_schedule={},
        attn_local_voxels=0,
        attn_local_1d_window=0,
        tea_cache_threshold=0.65,
        world_size=1
    )
    return cfgs

def generate_configs_wan() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    # Dit Fast Attention
    cfgs += make_config(
        base_path="examples/wan/chipmunk-config.yml",
        patchify=False,
        attn_sparsity=0,
        attn_full_step_every=1,
        attn_full_step_schedule={0, 1},
        attn_recompute_mask=True,
        mlp_sparsity=0,
        mlp_rk=0,
        mlp_mbm=0,
        mlp_is_fp8=False,
        mlp_full_step_every=1,
        mlp_block_mask_cache=0,
        step_caching=False,
        skip_step_schedule={},
        width=1280,
        height=720,
        global_disable_offloading=False,
        attn_local_voxels=0,
        attn_local_1d_window=0.1,
        world_size=1,
        attn_rk=0
    )
    # Tea Cache Config
    cfgs += make_config(
        base_path="examples/wan/chipmunk-config.yml",
        patchify=False,
        attn_sparsity=0.0,
        attn_full_step_every=1,
        attn_recompute_mask=False,
        mlp_sparsity=0,
        mlp_rk=0,
        mlp_mbm=0,
        mlp_is_fp8=False,
        mlp_full_step_every=1,
        mlp_block_mask_cache=0,
        step_caching=False,
        skip_step_schedule={},
        width=1280,
        height=720,
        global_disable_offloading=True,
        attn_full_step_schedule={},
        attn_local_voxels=0,
        attn_local_1d_window=0,
        tea_cache_threshold=0.2,
        world_size=1
    )
    # STA
    cfgs += make_config(
        base_path="examples/wan/chipmunk-config.yml",
        patchify=True,
        attn_sparsity=0,
        attn_full_step_every=1,
        attn_full_step_schedule={0, 1},
        attn_recompute_mask=True,
        mlp_sparsity=0,
        mlp_rk=0,
        mlp_mbm=0,
        mlp_is_fp8=False,
        mlp_full_step_every=1,
        mlp_block_mask_cache=0,
        step_caching=False,
        skip_step_schedule={},
        width=1280,
        height=720,
        global_disable_offloading=False,
        attn_local_voxels=5,
        attn_local_1d_window=0,
        world_size=1,
        attn_rk=0
    )
    # Chipmunk + TeaCache
    # cfgs += make_config(
    #     base_path="examples/wan/chipmunk-config.yml",
    #     patchify=True,
    #     attn_sparsity=0.1,
    #     attn_full_step_every=1,
    #     attn_full_step_schedule={0, 1, 10, 40},
    #     attn_recompute_mask=True,
    #     mlp_sparsity=0,
    #     mlp_rk=0,
    #     mlp_mbm=0,
    #     mlp_is_fp8=False,
    #     mlp_full_step_every=1,
    #     mlp_block_mask_cache=0,
    #     step_caching=False,
    #     skip_step_schedule={},
    #     width=1280,
    #     height=720,
    #     global_disable_offloading=False,
    #     attn_local_voxels=3,
    #     attn_local_1d_window=0,
    #     tea_cache_threshold=0.2,
    #     world_size=1,
    #     attn_rk=0.01
    # )
    return cfgs


def generate_configs(model_name: str) -> List[Dict[str, Any]]:  # noqa: D401,E501
    if model_name == "flux":
        return generate_configs_flux()
    elif model_name == "hunyuan":
        return generate_configs_hunyuan()
    elif model_name == "wan":
        return generate_configs_wan()
    else:
        raise NotImplementedError(
            f"generate_configs must be implemented by the user for model '{model_name}'"
        )


# -----------------------------------------------------------------------------
# Harness helpers
# -----------------------------------------------------------------------------

def _shortname_from_cfg(cfg: Dict[str, Any], idx: int) -> str:
    """Create a concise filesystem‑safe identifier for *cfg*."""

    keys: dict[str, str] = {}
    for p_key, p_val in cfg.items():
        if isinstance(p_val, dict):
            for c_key, c_val in p_val.items():
                if isinstance(c_val, (str, int, float)):
                    safe = str(c_val).replace("True", "T").replace("False", "F").replace("0.", ".")
                    safe = "".join(ch for ch in safe if ch.isalnum() or ch == ".")
                    keys[f"{p_key}.{c_key}"] = safe

    # friendly remaps for common hyper‑params
    remaps = {
        "mlp.top_keys": "mlptk",
        "attn.top_keys": "attntk",
        "mlp.full_step_every": "mlpfse",
        "attn.full_step_every": "attnfse",
        "attn.full_step_schedule": "attnfss",
        "attn.local_voxels": "attnlv",
        "attn.local_1d_window": "attnl1d",
        "attn.recompute_mask": "attnrm",
        "mlp.block_mask_cache": "mlpbmc",
        "step_caching.is_enabled": "stepcache",
        "mlp.random_keys": "mlprk",
        "mlp.mbm": "mlpmbm",
        "mlp.is_fp8": "mlpfp8",
        "width": "w",
        "height": "h",
    }

    short: list[str] = []
    for k_long, k_short in remaps.items():
        if k_long in keys:
            short.append(f"{k_short}={keys[k_long]}")
    
    if cfg['tea_cache']['is_enabled']:
        short.insert(0, f"teacache={cfg['tea_cache']['threshold']}")

    return "_".join(short)


def _ensure_dirs(base: Path, eval_name: str) -> None:
    """Create `media/` and `logs/` directories for *exp_dir* if missing."""

    (base / "media" / eval_name).mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)


def _launch_process(
    model: str,
    example_dir: Path,
    num_gpus: int,
    prompt_file: Path,
    cfg_path: Path,
    log_dir: Path,
) -> List[subprocess.Popen[bytes]]:
    """Spawn *num_gpus* processes and return their `Popen` handles."""

    procs: list[subprocess.Popen[bytes]] = []
    world_size = yaml.safe_load(open(cfg_path, 'r'))['world_size']
    num_processes = num_gpus // world_size

    for proc_id in range(num_processes):
        start_gpu = proc_id * world_size
        end_gpu = start_gpu + world_size

        cmd = [
            sys.executable,
            "batch_sample.py",
            "--prompt-file",
            str(prompt_file),
            "--chipmunk-config",
            str(cfg_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in range(start_gpu, end_gpu))
        env["CHIPMUNK_LOCAL_RANK"] = str(proc_id)
        env["CHIPMUNK_WORLD_SIZE"] = str(num_processes)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        stdout_f = (log_dir / f"gpu{proc_id}.out").open("w")
        stderr_f = (log_dir / f"gpu{proc_id}.err").open("w")
        
        # Pretty print command for copy-paste
        cmd_str = " ".join(cmd)
        print(f"\n[batch_harness] Command to run:")
        print("=" * 80)
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} CHIPMUNK_LOCAL_RANK={env['CHIPMUNK_LOCAL_RANK']} CHIPMUNK_WORLD_SIZE={env['CHIPMUNK_WORLD_SIZE']} \\\n{cmd_str}")
        print("=" * 80)
        
        p = subprocess.Popen(
            cmd,
            cwd=example_dir,
            stdout=stdout_f,
            stderr=stderr_f,
            bufsize=1,
            universal_newlines=False,
            env=env,
        )
        procs.append(p)

    return procs


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Chipmunk batch sampling harness")
    parser.add_argument(
        "--model",
        required=True,
        choices=["flux", "hunyuan", "mochi", "wan", "hidream"],
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--eval",
        required=True,
        choices=["vbench", "imre", "geneval"],
        help="Evaluation benchmark name.",
    )
    parser.add_argument(
        "--chipmunk-config",
        required=False,
        help="Path to an explicit Chipmunk config (skips grid).",
    )
    parser.add_argument(
        "--node-rank",
        required=False,
        default=0,
        type=int,   
        help="Node rank for multi-node inference.",
    )
    parser.add_argument(
        "--num-nodes",
        required=False,
        default=1,
        type=int,
        help="Number of nodes for multi-node inference.",
    )
    
    args = parser.parse_args(argv)

    args.num_nodes = int(args.num_nodes)
    args.node_rank = int(args.node_rank)

    model_name: str = args.model
    eval_name: str = args.eval

    prompt_file = PROMPT_DIR / f"{eval_name}.json"
    if not prompt_file.exists():
        sys.exit(f"Prompt file not found: {prompt_file}")

    # total number of prompts determines progress‑bar length
    with prompt_file.open() as fp:
        num_output_files = 0
        for generation in json.load(fp):
            if type(generation['output_path']) == list:
                num_output_files += len(generation['output_path'])
            else:
                num_output_files += 1

    import torch

    # ---------------------------------------------------------------------
    # Build configs (grid or user‑supplied)
    # ---------------------------------------------------------------------
    if args.chipmunk_config:
        with open(args.chipmunk_config, "r") as f:
            cfgs = [yaml.safe_load(f)]
    else:
        print(f"[batch_harness] generating configs for {model_name} …")
        cfgs = generate_configs(model_name)
        if not cfgs:
            sys.exit("generate_configs returned no configs – nothing to do.")
        print(f"[batch_harness] generated {len(cfgs)} configs for {model_name}.")

    # locate example code directory
    example_dir = ROOT_DIR / "examples" / EXAMPLE_DIR_MAP.get(model_name, model_name)
    if not example_dir.exists():
        sys.exit(f"Examples directory for model '{model_name}' not found: {example_dir}")

    # ---------------------------------------------------------------------
    # Iterate over configs
    # ---------------------------------------------------------------------
    num_gpus = torch.cuda.device_count()

    for idx, cfg in enumerate(cfgs):
        if idx % args.num_nodes != args.node_rank:
            continue

        shortname = _shortname_from_cfg(cfg, idx)
        exp_dir = MODEL_DIR / model_name / shortname
        _ensure_dirs(exp_dir, eval_name)

        if (exp_dir / "done.txt").exists():
            print(f"[batch_harness] skipping {shortname} (already done)")
            continue

        # write config to disk
        cfg_path = exp_dir / "chipmunk-config.yml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f)

        procs = _launch_process(
            model_name,
            example_dir,
            num_gpus,
            prompt_file.resolve(),
            cfg_path.resolve(),
            exp_dir / "logs",
        )

        media_dir = exp_dir / "media" / eval_name
        produced = 0
        # poll once per second until done
        while True:
            new_produced = sum(1 for _ in media_dir.rglob('*') if _.is_file()) if media_dir.exists() else 0
            if new_produced != produced:
                produced = new_produced
                print(f"\r{shortname}:{eval_name} [{produced}/{num_output_files}]", end="", flush=True)

            if produced >= num_output_files:
                break  # all prompts have outputs

            if all(p.poll() is not None for p in procs):
                break  # children finished (possibly early)

            time.sleep(1)

        print()  # final newline

        # final sanity: check exit codes
        for p in procs:
            code = p.wait()
            if code != 0:
                print(
                    f"[batch_harness] process for config '{shortname}' exited with code {code}"
                )
                sys.exit(code)

        # mark experiment directory as finished if all processes succeeded
        if all(p.wait() == 0 for p in procs) and len(procs) > 0:
            (exp_dir / "done.txt").write_text("finished\n")


if __name__ == "__main__":
    main()
