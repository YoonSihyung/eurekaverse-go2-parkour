# Setup — Distill a depth (student) policy from the seed-1 teacher

This repo is a fork of **extreme-parkour** + **Eurekaverse**, ported to **Isaac Lab 3.0**.
The included checkpoint is the **teacher** (privileged, scandots-based) policy. It is **not deployable
standalone** — it needs privileged terrain scandots that only exist inside the training sim.
To run the policy on a real robot / another sim, you must **distill** it into a **depth-based student**.

## Requirements (must be installed on the target machine)

- NVIDIA GPU (≈ 16–24 GB VRAM for depth distillation with 192 envs) + recent driver
- **Isaac Sim + Isaac Lab 3.0** (this is the large external dependency — a `git clone` does NOT include it)
- A conda env with this project's Python deps — see `environment_eureka_lab6.yml` / `pip_freeze_eureka_lab6.txt` (reference exports from the machine that trained the teacher)

> The exact simulator matters: the teacher was trained on **this Isaac Lab 3.0 port**. Distilling it
> requires the **same** env (same observation/scandots/control pipeline). A different sim version will
> not be compatible with the checkpoint.

## Setup

```bash
# 1) clone
git clone <this-repo-url> && cd <repo>

# 2) install Isaac Sim + Isaac Lab 3.0  (follow NVIDIA's installer for 3.0)

# 3) create the conda env and install this project (editable)
conda create -n eureka_lab6 python=3.12 -y && conda activate eureka_lab6
#   install isaaclab / isaacsim per NVIDIA docs, then:
pip install -e extreme-parkour/rsl_rl
pip install -e extreme-parkour/legged_gym
#   (see environment_eureka_lab6.yml for the full dependency set)
```

## Teacher checkpoint (included)

```
extreme-parkour/legged_gym/logs/parkour/2026-07-15_12-17-17_4_1/
  model_11000.pt            # teacher final policy (benchmark 4.36/8)
  legged_robot_config.pkl   # env/train config (required to load)
```

## Run distillation (teacher → depth student)

Phase-2 distillation renders the onboard depth camera and trains a depth encoder/actor to imitate the teacher:

```bash
cd extreme-parkour/legged_gym/legged_gym
export OMNI_KIT_ACCEPT_EULA=YES
# glib preload avoids an X11 startup segfault on some setups:
DEPS="$CONDA_PREFIX/lib/python3.12/site-packages/isaacsim/extscache/omni.gpu_foundation-*/bin/deps"
export LD_PRELOAD="$(ls $DEPS/libglib-2.0.so.0):$(ls $DEPS/libgobject-2.0.so.0)"

python scripts/train.py --task go2 --use_camera --resume \
    --load_run 2026-07-15_12-17-17_4_1 \
    --exptid   2026-07-15_12-17-17_4_1_distill \
    --headless
```

Notes:
- `--use_camera` switches to Phase-2 (depth) distillation (depth rendering, `depth_encoder.if_depth=True`).
- The student is written to `logs/parkour/2026-07-15_12-17-17_4_1_distill/`.
- Distillation on this Isaac Lab 3.0 port is **not yet fully validated** — expect to verify/adjust args.
- Evaluate the student with `scripts/evaluate.py ... --terrain_type benchmark` (see repo).

## Attribution / license

Built on **Eurekaverse** (Liang et al., CoRL 2024) and **extreme-parkour**. Preserve their upstream
licenses and attribution when redistributing.
