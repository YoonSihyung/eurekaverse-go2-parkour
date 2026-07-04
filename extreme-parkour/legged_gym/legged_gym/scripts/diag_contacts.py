"""Diagnostic: measure per-sensor contact forces while standing (zero actions)."""
import argparse, os, sys
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
args = parser.parse_args(); args.headless = True; args.script = "train"
app = AppLauncher(args).app
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry
env, _ = task_registry.make_env(args=args, name=args.task, render_mode=None)
print("BODY NAMES:", env._robot.body_names)
for step in range(60):
    env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    if step in (20, 40, 59):
        f = torch.norm(env.contact_forces_FOOT, dim=-1)   # (envs, 4)
        t = torch.norm(env.contact_forces_THIGH, dim=-1)
        c = torch.norm(env.contact_forces_CALF, dim=-1)
        print(f"step {step}: FOOT force mean={f.mean():.2f} max={f.max():.1f} | "
              f"THIGH mean={t.mean():.3f} contact%={(t>0.1).float().mean()*100:.1f} | "
              f"CALF mean={c.mean():.3f} contact%={(c>0.1).float().mean()*100:.1f}")
print("DIAG DONE")
app.close()
