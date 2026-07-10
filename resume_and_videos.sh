#!/usr/bin/env bash
# 루프 재개(iteration 11) → 완주 후 영상 배치 자동 실행. 세션 독립(setsid)이라 터미널/세션이 죽어도 계속 돈다.
# 사용법: ~/workspace/eurekaverse_lab3/resume_and_videos.sh
rm -f /dev/shm/sem.carbonite-sharedmemory /dev/shm/carb-RStringInternals-* /dev/shm/sem.carb-RStringInternals-*
setsid bash -c '
    source ~/.bashrc 2>/dev/null
    cd ~/workspace/eurekaverse_lab3/eurekaverse
    ~/miniconda3/envs/eureka_lab6/bin/python -u run_eurekaverse.py > ~/loop_production.log 2>&1
    echo "[$(date)] 루프 종료 (exit $?), 영상 배치 시작" >> ~/loop_production.log
    rm -f /dev/shm/sem.carbonite-sharedmemory
    bash ~/workspace/eurekaverse_lab3/make_videos.sh > ~/make_videos.log 2>&1
    echo "[$(date)] 영상 배치 종료 (exit $?)" >> ~/make_videos.log
' < /dev/null > /dev/null 2>&1 &
sleep 3
pgrep -f "python -u run_eurekaverse" > /dev/null && echo "루프 재개됨 (세션 독립). 확인: tail ~/loop_production.log" || echo "시작 실패 — 로그 확인 필요"
