#!/usr/bin/env bash
# Live-view training launcher: runs a SMALL training session with the kit viewer
# so you can watch the robots learn in real time. Independent of any background
# headless training (uses its own exptid; smaller envs/terrain to fit VRAM).
#
# Usage:
#   ./train_view.sh                # defaults: 512 envs, 2x4 terrain, 500 iters
#   ./train_view.sh 1000           # custom iteration count
set -e

ITERS="${1:-500}"

DEPS="$HOME/miniconda3/envs/eureka_lab6/lib/python3.12/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+6312fa25.lx64.r.cp312/bin/deps"
export LD_PRELOAD="$DEPS/libglib-2.0.so.0:$DEPS/libgobject-2.0.so.0"
export OMNI_KIT_ACCEPT_EULA=YES

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate eureka_lab6

cd "$(dirname "$0")/extreme-parkour/legged_gym/legged_gym"
python -u scripts/train.py --task go2 --exptid gui_watch \
    --max_iterations "$ITERS" --terrain_type simple \
    --num_envs 512 --num_rows 2 --num_cols 4 \
    --gui --viz kit
