# 직접 조작 가이드 (운전석 매뉴얼)

모든 경로는 `~/workspace/eurekaverse_lab3/` 기준. 학습 관련 명령은 먼저:
```bash
conda activate eureka_lab6
cd ~/workspace/eurekaverse_lab3/extreme-parkour/legged_gym/legged_gym
```

---

## 1. 학습 실행

```bash
python -u scripts/train.py --task go2 --exptid 실험이름 \
    --max_iterations 1000 --terrain_type simple --num_envs 6144 --headless
```

| 인자 | 의미 |
|---|---|
| `--exptid` | 실험 이름 = 체크포인트 폴더명 (`logs/parkour/실험이름/`). **실험마다 새 이름 사용** |
| `--max_iterations` | 학습 iteration 수 (1 iter ≈ 5초 @6144 envs) |
| `--terrain_type` | `simple`(평지+약간의 요철) / `benchmark`(20태스크) / `default`(GPT 지형) 등 |
| `--num_envs` | 병렬 로봇 수 (VRAM: 6144 ≈ 10GB) |
| `--num_rows N --num_cols M` | 지형 그리드 크기 오버라이드 (기본 10×40) |
| `--resume --load_run 이름` | 기존 체크포인트에서 이어서 학습 |
| `--seed N` | 랜덤 시드 |
| `--gui --viz kit` | 학습 과정을 창으로 보기 (느려짐; `train_view.sh` 참고) |

백그라운드로 걸기:
```bash
nohup python -u scripts/train.py ... > ~/mylog.log 2>&1 &
grep -E "Mean reward \(total\)|Mean episode length" ~/mylog.log | tail   # 진행 확인
```

## 2. 보상 조정 — [envs/base/legged_robot_config.py](../extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot_config.py)

**보상 가중치**: `RewardScales` 클래스 (443행 부근)
```python
class RewardScales:
    tracking_goal_vel = 1.5   # 골 방향 속도 (주 양성 보상)
    tracking_yaw = 0.5        # 골 방향 바라보기
    lin_vel_z = -1.0          # 수직 진동 페널티
    collision = -10.          # thigh/calf 접촉 페널티
    action_rate = -0.1        # 행동 급변 페널티
    ...                       # 0으로 두면 해당 항 비활성
```
- 관련 상수: `CustomRewardsCfg`(463행) — `tracking_sigma`, `base_height_target`, `only_positive_rewards`

**보상 함수 수식 자체**: [envs/base/legged_robot.py](../extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot.py)의 `_reward_<이름>` 메서드들 (파일 끝부분).
**새 보상 추가**: ① `RewardScales`에 `내보상 = 0.5` 추가 ② env에 `def _reward_내보상(self): return <num_envs 텐서>` 추가 — 자동으로 등록됨.

## 3. 그 외 주요 조정 지점 (같은 config 파일)

| 항목 | 위치 | 내용 |
|---|---|---|
| 로봇 (Go2) | `UNITREE_GO2_CFG` 79행 | Kp/Kd(stiffness/damping), 토크 한계, 초기 자세·높이 |
| 속도 명령 범위 | 330행 부근 `lin_vel_x = [0.3, 0.8]` | 커맨드 샘플 범위 (평지 사전학습용 주석 참고) |
| 도메인 랜덤화 | `CustomDomainRandCfg` 406행 | 마찰/질량/COM/모터/push 플래그와 범위 |
| 지형 | `CustomTerrainCfg` 259행 | 그리드, 스케일, 커리큘럼 on/off |
| 골/에피소드 | `CustomEnvCfg` (155행 부근) | `next_goal_threshold`, `reach_goal_delay`, `episode_length_s` |
| PPO 하이퍼파라미터 | `LeggedRobotCfgPPO` 532행 | learning_rate, entropy_coef(578행), num_steps_per_env 등 |

## 4. 평가·시각화

```bash
# 정량 평가 (골 도달 수가 핵심 지표)
python -u scripts/evaluate.py --task go2 --exptid 실험이름 --headless \
    --max_steps 1001 --metric_granularity type --terrain_type simple

# 학습된 정책 구경 (GUI)
~/workspace/eurekaverse_lab3/view.sh 실험이름

# 학습 과정 실시간 관람 (소규모 학습 + GUI)
~/workspace/eurekaverse_lab3/train_view.sh
```

## 5. 실험 워크플로 권장 패턴

```bash
cd ~/workspace/eurekaverse_lab3
git diff                      # 지금 뭘 바꿨는지 확인
# (config 수정) → 새 exptid로 학습 → 평가 → 비교
git stash                     # 실험 변경 임시 제거 (되돌리기)
git checkout -- .             # 검증된 baseline(cc1216c)으로 완전 복귀
git add -A && git commit -m "실험: xxx"   # 유의미한 변경은 커밋
```

⚠️ **주의**: 현재 config 값들은 논문 원본과의 전수 대조로 검증된 상태입니다 ([verification_report.md](verification_report.md)). 논문 재연용 학습(walk_pretrain, 루프)은 baseline 값으로 돌려야 하므로, 개인 실험은 **별도 브랜치나 커밋으로 구분**하는 것을 권장합니다:
```bash
git checkout -b my-experiments    # 실험용 브랜치
git checkout master               # 재연용 baseline 복귀
```

## 6. 파일 지도

```
extreme-parkour/legged_gym/legged_gym/
├── envs/base/legged_robot_config.py  ← 거의 모든 파라미터 (보상/DR/지형/PPO/로봇)
├── envs/base/legged_robot.py         ← 환경 로직 (보상 함수, 관측, 골, 커리큘럼, DR 적용)
├── envs/__init__.py                  ← 태스크 등록 (go1/go2)
├── utils/terrain_gpt.py              ← 지형 생성·trimesh 임포트 (청크 분할 포함)
├── utils/set_terrain*.py             ← 지형 정의 (simple/benchmark/...)
├── scripts/train.py, evaluate.py     ← 실행 진입점
└── logs/parkour/<exptid>/            ← 체크포인트(model_*.pt), 설정 스냅샷
extreme-parkour/rsl_rl/               ← PPO/네트워크 (ActorCriticRMA, estimator)
eurekaverse/                          ← LLM 커리큘럼 루프 (연결 작업 중)
docs/reproduction_spec.md             ← 원본 논문 스펙 (수치의 근거)
docs/verification_report.md           ← 원본 대비 검증 내역
```
