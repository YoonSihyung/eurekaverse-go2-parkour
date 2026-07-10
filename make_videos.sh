#!/usr/bin/env bash
# 재현 스토리 영상 세트 생성. GPU가 비어 있을 때 실행 (루프와 동시 실행 금지).
#
# 산출물: ~/workspace/eurekaverse_lab3/videos/
#   1) learning/it-N_{pre|post}_row{R}.mp4 — iteration별 지형에서 학습 전(부모 정책) vs 후. 지형이 점점 어려워지는 걸 보여줌
#   2) benchmark/<정책>_task{T}.mp4        — 같은 벤치마크 코스에서 정책 세대별 발전
set -e
cd "$(dirname "$0")/extreme-parkour/legged_gym/legged_gym"

DEPS="$HOME/miniconda3/envs/eureka_lab6/lib/python3.12/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+6312fa25.lx64.r.cp312/bin/deps"
export LD_PRELOAD="$DEPS/libglib-2.0.so.0:$DEPS/libgobject-2.0.so.0"
export OMNI_KIT_ACCEPT_EULA=YES
PY=~/miniconda3/envs/eureka_lab6/bin/python
RUN=2026-07-06_12-22-44
OUT=~/workspace/eurekaverse_lab3/videos
LOG=~/loop_production.log
mkdir -p $OUT/learning $OUT/benchmark

# 루프 로그에서 iteration N의 승자 run id 추출 (없으면 빈 값)
winner() { grep -oE "Best run in iteration $1 is run [0-9]+" $LOG ~/loop_production*.log 2>/dev/null | tail -1 | grep -oE "[0-9]+$"; }

# 학습지형 영상: eval(정책, 지형, 라벨) — 난이도 중(row3)/최고(row7) 카메라 2개
learn_vid() { # $1=exptid(정책) $2=terrain_type $3=라벨
    echo "=== learning: $3 ($1 on $2) ==="
    $PY -u scripts/evaluate.py --task go2 --exptid "$1" --headless --video \
        --terrain_type "$2" --video_row_idxes 3,7 --num_rows 8 --num_cols 1 \
        --num_envs 8 --num_terrain_types 1 --max_steps 600 --no_save 2>&1 | tail -2
    for f in ../logs/parkour/$1/eval_videos/cam_*.mp4; do
        row=$(basename $f | grep -oE "^cam_[0-9]+" | grep -oE "[0-9]+")
        mv "$f" "$OUT/learning/$3_row${row}.mp4"
    done
}

# 벤치마크 영상: 태스크 5종(램프/높은박스/디딤돌/계단/장대), 중간 난이도(row4)
bench_vid() { # $1=exptid $2=라벨
    echo "=== benchmark: $2 ($1) ==="
    $PY -u scripts/evaluate.py --task go2 --exptid "$1" --headless --video \
        --terrain_type benchmark --video_row_idxes 4 --num_rows 8 --num_cols 5 \
        --num_envs 40 --num_terrain_types 5 --max_steps 600 --no_save 2>&1 | tail -2
    i=0; tasks=(ramp highbox stones stairs poles)
    for f in $(ls ../logs/parkour/$1/eval_videos/cam_*.mp4 | sort -t c -k3 -n); do
        mv "$f" "$OUT/benchmark/$2_${tasks[$i]}.mp4"; i=$((i+1))
    done
}

W10=$(winner 10); W11=$(winner 11)

# ── 1. iteration별 pre→post (지형 난이도 진화 + 학습 증거) ──
learn_vid ${RUN}_0_0 it-1_run-0 "it01_pre";  learn_vid ${RUN}_1_0 it-1_run-0 "it01_post"
learn_vid ${RUN}_4_0 it-5_run-0 "it05_pre";  learn_vid ${RUN}_5_0 it-5_run-0 "it05_post"
learn_vid ${RUN}_7_0 it-8_run-3 "it08_pre";  learn_vid ${RUN}_8_3 it-8_run-3 "it08_post"
if [ -n "$W10" ]; then
    learn_vid ${RUN}_9_0 it-10_run-$W10 "it10_pre"; learn_vid ${RUN}_10_$W10 it-10_run-$W10 "it10_post"
fi
if [ -n "$W11" ]; then
    learn_vid ${RUN}_10_${W10:-0} it-11_run-$W11 "it11_pre"; learn_vid ${RUN}_11_$W11 it-11_run-$W11 "it11_post"
fi

# ── 2. 벤치마크 발전사 (같은 시험 코스, 정책 세대별) ──
bench_vid walk_pretrain "gen0_pretrain"
bench_vid ${RUN}_1_0    "gen1_it1"
bench_vid ${RUN}_7_0    "gen2_it7"
bench_vid ${RUN}_8_3    "gen3_it8"
[ -n "$W10" ] && bench_vid ${RUN}_10_$W10 "gen4_it10"
[ -n "$W11" ] && bench_vid ${RUN}_11_$W11 "gen5_it11"

echo "완료: $(ls $OUT/learning | wc -l) learning + $(ls $OUT/benchmark | wc -l) benchmark 영상 → $OUT"
