"""Diagnostic: spawn robots with zero actions and report why they terminate.

Used to debug the horizontal_scale=0.05 instant-death issue. Not part of the pipeline.
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.script = "train"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry

env, env_cfg = task_registry.make_env(args=args, name=args.task, render_mode=None)

print(f"DIAG: num_envs={env.num_envs}, horizontal_scale={env_cfg.terrain.horizontal_scale}")
origins = getattr(env, "env_origins", None)
if origins is None:
    origins = env.initial_env_origins
print(f"DIAG: env_origins z: min={origins[:,2].min():.3f} max={origins[:,2].max():.3f}")

root = env._robot.data.root_state_w
spawn_z = root[:, 2].clone()
print(f"DIAG: spawn root z: min={root[:,2].min():.3f} mean={root[:,2].mean():.3f} max={root[:,2].max():.3f}")

for step in range(40):
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    env.step(actions)
    root = env._robot.data.root_state_w
    roll, pitch = env.roll, env.pitch
    n_reset = int(env.reset_buf.sum())
    if step % 5 == 0 or n_reset > env.num_envs * 0.3:
        print(f"DIAG step {step:3d}: z_mean={root[:,2].mean():.3f} z_min={root[:,2].min():.3f} "
              f"|roll|_max={roll.abs().max():.2f} |pitch|_max={pitch.abs().max():.2f} resets={n_reset}")
        # termination cause breakdown (mirrors check_termination)
        r_roll = (roll.abs() > 1.5).sum().item()
        r_pitch = (pitch.abs() > 1.5).sum().item()
        print(f"          cause: roll>{1.5}: {r_roll}  pitch>{1.5}: {r_pitch}")

print("DIAG DONE")
simulation_app.close()
