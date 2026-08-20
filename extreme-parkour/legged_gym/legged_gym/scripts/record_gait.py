"""Record a REAL walk gait pattern from a trained policy (e.g. walk_pretrain) by logging foot contacts.

Runs the policy in its OWN validated env (so the gait is faithful) and folds the steady-state foot-
contact signal into a 4xT 0/1 step pattern (the WHEN). Used to ground the saytap step-pattern library
in a real controller. Modeled on scripts/diag_contacts.py (env + contact_forces_FOOT) + the policy
loading in scripts/evaluate.py.

  ./view.sh-style env, headless:
    python scripts/record_gait.py --task go2 --exptid walk_pretrain --num_envs 4 --headless --secs 6
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
parser.add_argument("--secs", type=float, default=6.0)
parser.add_argument("--T", type=int, default=24)
parser.add_argument("--vx", type=float, default=0.6, help="forced forward velocity command")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.script = "play"
app = AppLauncher(args).app

import numpy as np
import torch
from pathlib import Path
from legged_gym.envs import *  # noqa: F401,F403  (registers tasks)
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import task_registry


def extract_pattern(C, T=24):
    """C: (N,4) 0/1 foot contacts at control rate -> 4xT 0/1 pattern via period detection + folding."""
    C = np.asarray(C, float)
    N = len(C)
    ref = C[:, 0] - C[:, 0].mean()
    best_p, best_v = 8, -1e9
    for p in range(8, min(80, N // 2)):
        v = float(np.dot(ref[:-p], ref[p:]) / (N - p))
        if v > best_v:
            best_v, best_p = v, p
    P = best_p
    folded = np.zeros((P, 4)); cnt = np.zeros(P)
    for i in range(N):
        folded[i % P] += C[i]; cnt[i % P] += 1
    folded = (folded / np.maximum(cnt[:, None], 1)) > 0.5
    pat = np.zeros((4, T), dtype=int)
    for c in range(T):
        pat[:, c] = folded[int(round(c / T * P)) % P].astype(int)
    return pat, P


def main():
    load_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / args.proj_name / args.exptid
    try:
        env_cfg, train_cfg = task_registry.get_saved_cfgs(load_dir=load_dir)
    except Exception:
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env, env_cfg = task_registry.make_env(args=args, name=args.task, env_cfg=env_cfg, render_mode=None)
    ppo_runner, train_cfg, _, loaded_dir, checkpoint = task_registry.make_alg_runner(
        env=env, args=args, name=args.task, train_cfg=train_cfg, log_root=load_dir)
    policy = ppo_runner.get_inference_policy(device=env.device)
    print("[gait] FOOT body order:", [env._robot.body_names[i] for i in env.feet_indices])

    obs = env.get_observations()
    if isinstance(obs, (tuple, list)):
        obs = obs[0]
    n = int(args.secs / env.dt)
    contacts, base_x, heights = [], [], []
    for t in range(n):
        env.commands[:, 0] = args.vx                            # force a forward velocity command
        with torch.no_grad():
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=None)
        all_obs, _, _, _, _ = env.step(actions.detach())
        obs = all_obs[0] if isinstance(all_obs, (tuple, list)) else all_obs
        f = torch.norm(env.contact_forces_FOOT[0], dim=-1)      # (4,)
        contacts.append((f > 2.0).cpu().numpy().astype(int))
        base_x.append(float(env._robot.data.root_pos_w[0, 0]))
        heights.append(float(env._robot.data.root_pos_w[0, 2]))

    C = np.array(contacts); heights = np.array(heights)
    fwd = base_x[-1] - base_x[0]
    print(f"[gait] steps={n} dt={env.dt:.4f} forward={fwd:.2f}m speed={fwd/args.secs:.3f}m/s "
          f"height[{heights.min():.3f},{heights.max():.3f}] duty_raw={C.mean():.2f}")
    n0 = len(C) // 3
    pat, P = extract_pattern(C[n0:], T=args.T)
    print(f"[gait] period={P} ticks ({P*env.dt:.2f}s) duty={pat.mean():.2f}")
    print("[gait] recorded 4xT pattern (rows = FOOT order above):")
    for i in range(4):
        print(f"[gait]   {i}: {''.join(str(x) for x in pat[i])}")
    out = Path("/home/yoonsihyung/workspace/cap_quad/saytap/artifacts") / f"{args.exptid}_pattern.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pat)
    print(f"[gait] saved {out}")
    app.close()


if __name__ == "__main__":
    main()
