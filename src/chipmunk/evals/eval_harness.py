from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT_DIR / "evals"
MODEL_ROOT = EVAL_ROOT / "models"

IMAGE_MODELS = {"flux", "hidream"}


def _launch(cmd: List[str]) -> int:
    """Run *cmd* synchronously inheriting stdout/stderr, return returncode."""
    print(f"[eval_harness] running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    proc.wait()
    return proc.returncode


def main(argv: List[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Chipmunk evaluation harness")
    parser.add_argument("--prefix", default="", help="Optional prefix under evals/models/ to narrow the search scope.")
    args = parser.parse_args(argv)

    search_root = MODEL_ROOT / args.prefix
    if not search_root.exists():
        sys.exit(f"Search path not found: {search_root}")

    exp_dirs: List[Path] = []
    for done_file in search_root.rglob("done.txt"):
        exp_dirs.append(done_file.parent)

    if not exp_dirs:
        print("[eval_harness] No completed experiments found.")
        return

    for exp_dir in exp_dirs:
        # Determine model name from path parts: .../models/<model>/<config>
        try:
            idx = exp_dir.parts.index("models")
            model_name = exp_dir.parts[idx + 1]
        except (ValueError, IndexError):
            print(f"[eval_harness] could not parse model name from {exp_dir}")
            continue

        media_dir = exp_dir / "media"
        evals_dir = exp_dir
        evals_dir.mkdir(exist_ok=True)

        is_image_model = model_name in IMAGE_MODELS
        cmds: List[List[str]] = []

        if is_image_model:
            # IMRE
            imre_out = evals_dir / "imre.json"
            if not imre_out.exists():
                cmds.append([
                    sys.executable,
                    "-m",
                    "chipmunk.evals.imre",
                    "--experiment-dir",
                    str(media_dir.resolve()),
                    "--out-path",
                    str(imre_out.resolve()),
                ])
            else:
                print(f"[eval_harness] IMRE already exists for {exp_dir}")
            # Geneval
            # geneval_out = evals_dir / "geneval.json"
            # if not geneval_out.exists():
            #     cmds.append([
            #         sys.executable,
            #         "-m",
            #         "chipmunk.evals.geneval",
            #         "--experiment-dir",
            #         str(exp_dir.resolve()),
            #         "--out-path",
            #         str(geneval_out.resolve()),
            #     ])
        else:
            vbench_out = evals_dir / "vbench"
            if not vbench_out.exists():
                import torch
                dimensions = [
                    "object_class",
                    "multiple_objects",
                    "human_action", 
                    "color",
                    "spatial_relationship",
                    "scene",
                    "appearance_style",
                    "temporal_style",
                    "overall_consistency",
                    "subject_consistency",
                    "background_consistency",
                    "temporal_flickering",
                    "motion_smoothness",
                    "aesthetic_quality",
                    "imaging_quality",
                    "dynamic_degree"
                ]
                for dimension in dimensions:
                    cmds.append([
                        "vbench",
                        "evaluate",
                        f"--ngpus={torch.cuda.device_count()}",
                        "--dimension",
                        dimension,
                        "--videos_path",
                        str(media_dir),
                        '--output_path',
                        str(vbench_out.resolve()),
                    ])

        for cmd in cmds:
            ret = _launch(cmd)
            if ret != 0:
                print(f"[eval_harness] Command failed with code {ret}: {' '.join(cmd)}")


if __name__ == "__main__":  # pragma: no cover
    main() 