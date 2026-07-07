#!/usr/bin/env bash
# 루프 진행 상황 한눈에 보기. 사용법: ./status.sh  (아무 터미널에서나, 실행에 영향 없음)
OUT=$(ls -td ~/workspace/eurekaverse_lab3/eurekaverse/outputs/run_eurekaverse/*/ 2>/dev/null | head -1)
# resume된 런이면 원래 런 폴더로 따라감
if [ -f "$OUT/resumed_run.txt" ]; then
    OUT=~/workspace/eurekaverse_lab3/eurekaverse/outputs/run_eurekaverse/$(cat "$OUT/resumed_run.txt")/
fi
LOOP=~/loop_production.log

echo "══════════ Eurekaverse 루프 상태 ══════════"
echo "출력 폴더: $OUT"

# 현재 iteration / 단계
IT=$(grep -oE "ITERATION [0-9]+" "$LOOP" 2>/dev/null | tail -1)
PHASE=$(grep -E "Starting .* subprocess|Querying OpenAI|Generated 10|Best run" "$LOOP" 2>/dev/null | tail -1 | sed 's/.*INFO\] - //')
echo "현재:      ${IT:-시작 전} — ${PHASE:-대기}"

# 학습 중이면 진행도
TRAIN=$(ls -t "$OUT"train_iter-*.log 2>/dev/null | head -1)
if [ -n "$TRAIN" ]; then
    N=$(grep -cE "Learning iteration" "$TRAIN")
    TOTAL=$(grep -oE "\-\-max_iterations [0-9]+" "$TRAIN" | head -1 | grep -oE "[0-9]+")
    echo "학습:      $(basename $TRAIN): $N / ${TOTAL:-?} iterations"
    grep -E "Mean reward \(total\)|Mean episode length|ETA" "$TRAIN" | tail -3 | sed 's/^ */  /'
fi

# 지금까지의 iteration별 벤치마크(testing) 성적
echo "── iteration별 testing 골 도달 (벤치마크, /8) ──"
for f in "$OUT"eval_iter-*_testing.log; do
    [ -f "$f" ] || continue
    G=$(grep -m1 "Number of goals reached" "$f" | grep -oE "[0-9.]+")
    echo "  $(basename $f .log): ${G:-집계 전}"
done

# 자원
echo "── 자원 ──"
free -h | awk 'NR==2{print "  RAM:", $3, "/", $2} NR==3{print "  스왑:", $3, "/", $2}'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | awk -F, '{print "  GPU:" $1 " 사용률" $2}'
echo "── 비용 ──"
grep -oE "cost: \\\$[0-9.]+" "$LOOP" 2>/dev/null | grep -oE "[0-9.]+" | awk '{s+=$1} END {printf "  API 누적: $%.2f\n", s}'
