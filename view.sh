#!/usr/bin/env bash
# GUI viewer launcher for the eurekaverse_lab3 stack.
#
# Isaac Sim 6.0's kit viewer needs its bundled GLib (2.87) loaded before the
# system one (2.72, missing g_dir_unref) — otherwise the GPU Foundation plugin
# fails and the app crashes. This wrapper preloads the bundled pair and runs
# evaluate.py with the kit viewer enabled.
#
# Usage:
#   ./view.sh <exptid> [extra evaluate.py args...]
# Examples:
#   ./view.sh go2_smoke
#   ./view.sh walk_pretrain --terrain_type benchmark --num_rows 4 --num_cols 4
set -e

EXPTID="${1:?usage: ./view.sh <exptid> [extra args...]}"
shift || true

DEPS="$HOME/miniconda3/envs/eureka_lab6/lib/python3.12/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+6312fa25.lx64.r.cp312/bin/deps"
export LD_PRELOAD="$DEPS/libglib-2.0.so.0:$DEPS/libgobject-2.0.so.0"
export OMNI_KIT_ACCEPT_EULA=YES

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate eureka_lab6

cd "$(dirname "$0")/extreme-parkour/legged_gym/legged_gym"
python -u scripts/evaluate.py --task go2 --exptid "$EXPTID" \
    --max_steps 5000 --metric_granularity type --terrain_type simple \
    --num_rows 4 --num_cols 4 --viz kit "$@"
