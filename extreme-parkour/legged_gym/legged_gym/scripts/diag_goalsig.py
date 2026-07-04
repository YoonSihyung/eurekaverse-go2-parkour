"""Diagnostic: verify goal-tracking observation channels are geometrically correct."""
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
e = 0  # env 0 관찰
for step in range(30):
    env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    if step % 10 == 9:
        root = env._robot.data.root_state_w[e]
        goal = env.cur_goals[e]
        rel = goal[:2] - root[:2]
        import math
        true_yaw_to_goal = math.atan2(rel[1].item(), rel[0].item())
        print(f"step {step}: root_xy=({root[0]:.2f},{root[1]:.2f}) goal_xy=({goal[0]:.2f},{goal[1]:.2f}) "
              f"dist={rel.norm():.2f}")
        print(f"  기하학적 골 방향={true_yaw_to_goal:.3f} | env.target_yaw={env.target_yaw[e]:.3f} "
              f"| env.yaw={env.yaw[e]:.3f} | delta_yaw={env.delta_yaw[e]:.3f} | cmd_velx={env.commands[e,0]:.2f} "
              f"| cur_goal_idx={env.cur_goal_idx[e].item()}")
        # 쿼터니언 순서 확인
        q = env._robot.data.root_quat_w[e]
        print(f"  root_quat_w={q.cpu().numpy().round(3)} (Isaac Lab 규약: w-first, 정지 시 [1,0,0,0] 근처여야 함)")
print("command_ranges:", env.command_ranges)
print("DIAG DONE")
app.close()
