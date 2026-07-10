#!/usr/bin/env bash
# 루프 완주 후 보충 영상: iteration 11 경계 쌍 + 벤치마크 중간 세대.
# (경계 1~3 쌍은 videos/curriculum_story/, worst/best 비교는 videos/benchmark_compare/에 이미 있음)
set -e
cd "$(dirname "$0")/extreme-parkour/legged_gym/legged_gym"

DEPS="$HOME/miniconda3/envs/eureka_lab6/lib/python3.12/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+6312fa25.lx64.r.cp312/bin/deps"
export LD_PRELOAD="$DEPS/libglib-2.0.so.0:$DEPS/libgobject-2.0.so.0"
export OMNI_KIT_ACCEPT_EULA=YES
PY=~/miniconda3/envs/eureka_lab6/bin/python
RUN=2026-07-06_12-22-44
OUT=~/workspace/eurekaverse_lab3/videos
mkdir -p $OUT/curriculum_story $OUT/benchmark_compare $OUT/terrain_progression
shopt -s nullglob

winner() { grep -oE "Best run in iteration $1 is run [0-9]+" ~/loop_production*.log 2>/dev/null | tail -1 | grep -oE "[0-9]+$"; }

# 학습지형 쌍: 정책을 지형 위에 올려 관찰 (row3=중간, row7=최고 난이도 / type 2종)
rec_terrain() { # $1=policy $2=terrain_type $3=label
    echo "=== $3 ==="
    $PY -u scripts/evaluate.py --task go2 --exptid "$1" --headless --video \
        --terrain_type "$2" --video_row_idxes 3,6 --num_rows 8 --num_cols 2 \
        --num_envs 16 --num_terrain_types 2 --max_steps 600 --no_save 2>&1 | tail -1
    for f in ../logs/parkour/$1/eval_videos/cam_*.mp4; do
        b=$(basename "$f" .mp4); row=${b#cam_}; row=${row%%r_*}; col=${b##*_c}
        mv "$f" "$OUT/curriculum_story/${3}_row${row}_type${col}.mp4"
    done
}

# 벤치마크 세대: 시험 코스 5종 (ramp/highbox/stones/stairs/poles), easy+hard
rec_bench() { # $1=policy $2=label
    echo "=== benchmark: $2 ==="
    $PY -u scripts/evaluate.py --task go2 --exptid "$1" --headless --video \
        --terrain_type benchmark --video_row_idxes 1,7 --num_rows 8 --num_cols 5 \
        --num_envs 40 --num_terrain_types 5 --max_steps 600 --no_save 2>&1 | tail -1
    tasks=(ramp highbox stones stairs poles)
    for f in ../logs/parkour/$1/eval_videos/cam_*.mp4; do
        b=$(basename "$f" .mp4); row=${b#cam_}; row=${row%%r_*}; col=${b##*_c}
        dl=easy; [ "$row" = "7" ] && dl=hard
        mv "$f" "$OUT/benchmark_compare/${2}_${tasks[$col]}_${dl}.mp4"
    done
}

W11=$(winner 11)

# ── 서사 1 보충: 앵커+폭8 경계 (it10 마스터가 it11 지형에서 고전) ──
if [ -n "$W11" ]; then
    rec_terrain ${RUN}_11_$W11 it-11_run-$W11 "mastery6_it11정책_it11지형"
    rec_terrain ${RUN}_10_3 it-11_run-$W11 "struggle3_it10정책_it11지형"
    # 지형 진화 스틸에 it-11 추가
    cd ~/workspace/eurekaverse_lab3
    $PY render_terrains.py eurekaverse/outputs/run_eurekaverse/$RUN/terrain_iter-11_run-$W11.py > /dev/null 2>&1 \
        && mv eurekaverse/outputs/run_eurekaverse/$RUN/terrain_iter-11_run-$W11.png $OUT/terrain_progression/iteration_11.png
    cd extreme-parkour/legged_gym/legged_gym
fi

# ── 서사 2 보충: 벤치마크 중간·최종 세대 (worst 1.05, best 4.48은 이미 있음) ──
rec_bench ${RUN}_1_0 "mid3.69_it1"
[ -n "$W11" ] && rec_bench ${RUN}_11_$W11 "final_it11"

echo "완료. learning=$(ls $OUT/curriculum_story | wc -l), benchmark=$(ls $OUT/benchmark_compare | wc -l)"
