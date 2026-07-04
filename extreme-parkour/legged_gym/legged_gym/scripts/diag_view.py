"""Diagnostic: run the env with the kit viewer and capture viewport screenshots.

Used to debug 'empty viewport' reports. Not part of the pipeline.
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import AppLauncher
from legged_gym.utils import add_shared_args

parser = argparse.ArgumentParser()
add_shared_args(parser)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str)
parser.add_argument("--checkpoint", type=int, default=-1)
parser.add_argument("--max_iterations", type=int)
parser.add_argument("--shot_dir", type=str, default="/tmp/viewport_shots")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.script = "train"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry

env, env_cfg = task_registry.make_env(args=args, name=args.task, render_mode=None)
os.makedirs(args.shot_dir, exist_ok=True)

for step in range(150):
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    env.step(actions)
    if step in (30, 100):
        try:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            vp = get_active_viewport()
            print(f"DIAG viewport: {vp}, camera={vp.camera_path if vp else None}")
            capture_viewport_to_file(vp, os.path.join(args.shot_dir, f"shot_{step:03d}.png"))
            print(f"DIAG captured shot_{step:03d}.png")
        except Exception as e:
            print(f"DIAG capture failed: {type(e).__name__}: {e}")
# let capture writers flush
import time
for _ in range(30):
    simulation_app.update()
    time.sleep(0.05)
print("DIAG DONE")
simulation_app.close()
