# Eurekaverse 재현 스펙 (Ground Truth)

원본: `~/workspace/eurekaverse` (eureka-research/eurekaverse, Isaac Gym + Go1).
목표: Isaac Lab 3.0(v3.0.0-beta2.patch1) + Isaac Sim 6.0.0.1 + **Go2**로 동일 재현.
이 문서의 모든 수치는 원본 코드에서 직접 추출한 값이며, 구현·검증의 단일 기준이다.
원본 파일 경로 표기는 `extreme-parkour/legged_gym/legged_gym/` 기준 상대경로.

---

## 1. 관측 공간 (총 698차원)

출처: `envs/base/legged_robot_config.py:37-46`, `envs/base/legged_robot.py:412-468` (`compute_observations`)

`obs_buf = [proprio(48), scandots(132), priv_explicit(9), priv_latent(29), history(10×48)]`

### 1.1 proprio (n_proprio = 48) — 배포 시 실기에서 재구성해야 하는 부분
| # | 항목 | 차원 | 비고 |
|---|---|---|---|
| 1 | base 각속도 | 3 | `obs_scales.ang_vel` 스케일 |
| 2 | roll, pitch (IMU) | 2 | yaw 제외 |
| 3 | delta_yaw, delta_next_yaw | 2 | 현재/다음 골 방향과 heading 차이. **5스텝마다 재계산** (`legged_robot.py:417-419`). 히스토리 저장 시 0으로 마스킹 (`:451-459`) |
| 4 | 명령 lin_vel_x | 1 | |
| 5 | (dof_pos − default) | 12 | `obs_scales.dof_pos` |
| 6 | dof_vel | 12 | `obs_scales.dof_vel` |
| 7 | 직전 action | 12 | |
| 8 | 발 접촉 | 4 | `contact_filt − 0.5` |

### 1.2 scandots (n_scan = 132)
- 격자: `measured_points_x` 12개 × `measured_points_y` 11개 = 132 (`legged_robot_config.py:179-180`)
- 값: `root_z − 0.3 − measured_height`, clip [−1, 1] (`legged_robot.py:444-446`)

### 1.3 priv_explicit (n_priv = 9)
- base 선속도(3, 스케일 적용) + 0벡터(3) + 0벡터(3) (`legged_robot.py:435-437`)
- 추론 시 Estimator 출력으로 대체됨 (rsl_rl `ppo.py:146-149`)

### 1.4 priv_latent (n_priv_latent = 29)
- 질량 파라미터(4) + 마찰(1) + 모터 Kp 계수(12) + 모터 Kd 계수(12) (`legged_robot.py:438-443`)

### 1.5 history: 최근 10스텝 proprio (yaw 마스킹된 버전), `history_len=10`

## 2. 행동 공간 / 제어

| 항목 | 값 | 출처 |
|---|---|---|
| 액션 | 12 (관절 목표각 오프셋) | |
| 제어 | PD, `torque = motor_kp_factor·Kp·(a·scale + default − q) − motor_kd_factor·Kd·q̇` | `legged_robot.py:635-670` |
| Kp / Kd (Go1) | 20.0 / 0.5 | `go1_config.py:75-87` → **Go2 값은 §9** |
| action_scale | 0.25 | |
| sim dt / decimation | 0.005 / 4 → 정책 50Hz | `legged_robot_config.py:370` |
| action clip | ±1.2/action_scale | `legged_robot_config.py:135` |
| 토크 한계 | URDF/USD에서 읽음 (`_process_dof_props`) | `legged_robot.py:533-559` |

## 3. 보상 (14항, `only_positive_rewards=True`, `tracking_sigma=0.2`)

출처: `legged_robot_config.py:335-361`, 함수 `legged_robot.py:1361-1435`. 스케일은 dt 곱해짐.

| 항 | 스케일 | 핵심 로직 |
|---|---|---|
| tracking_goal_vel | **1.5** | 골 방향 투영 속도, 명령 속도로 캡 (`:1361-1379`) |
| tracking_yaw | **0.5** | exp(−|delta_yaw|/σ) |
| lin_vel_z | −1.0 | 평지(env_class==−1)에서만 활성 (`:1387`) |
| ang_vel_xy | −0.05 | |
| orientation | −1.0 | 평지에서만 (`:1395`) |
| dof_acc | −2.5e-7 | |
| collision | −10.0 | thigh/calf 접촉 페널티 |
| action_rate | −0.1 | |
| delta_torques | −1.0e-7 | |
| torques | −1e-5 | |
| hip_pos | −0.5 | |
| dof_error | −0.04 | |
| feet_stumble | −1 | |
| feet_edge | −1 | trimesh 변환 시 얻는 `x_edge_mask` 사용, terrain_level>3에서만 (`:1426-1434`) |

## 4. 골 추종 로직 (`_update_goals`, `legged_robot.py:222-243`)

- 골 8개/지형 (`num_goals=8`)
- 도달: 수평거리 < `next_goal_threshold=0.2m`가 `reach_goal_delay=0.1s` 지속 → `cur_goal_idx += 1`
- `target_yaw` = atan2(골 방향), delta_yaw 관측에 사용
- 종료(§5)와 커리큘럼(§6)이 `cur_goal_idx`를 소비

## 5. 종료 조건 (`check_termination`, `legged_robot.py:304-319`)

- |roll| > 1.5 rad, |pitch| > 1.5 rad, root z 드롭 < −0.25m
- 타임아웃: `episode_length_s=20` 초과 **또는** 8골 전부 도달 (실패 아님)

## 6. 지형 커리큘럼 (`_update_terrain_curriculum`, `legged_robot.py:726-773`)

- 골 기반: `cur_goal_idx ≥ 0.8×8` → 레벨 상승, `< 0.4×8` → 하강
- 정체(no_move) env는 25% 확률로 레벨 랜덤화
- 최고 레벨 도달 시 낮은 레벨로 랜덤 재배치
- 초기 분포: `max_init_terrain_level=2`

## 7. 지형 파이프라인 (`utils/terrain_gpt.py`)

### 7.1 그리드
| 항목 | 값 |
|---|---|
| 그리드 | `num_rows=10`(난이도) × `num_cols=40`(변형) — 원본 config 기준. 참고: 포팅 구현체는 8×20으로 축소 운용 |
| 셀 크기 | 18m × 4m (`terrain_length × terrain_width`) |
| horizontal_scale / vertical_scale | 0.05m / 0.005m |
| border_size / slope_threshold | 5m / 1.5 |
| difficulty / variation | `row/(num_rows−1)`, `col/num_cols` |
| 셀별 시드 | `int(variation*1e3 + difficulty*1e6)` (`terrain_gpt.py:122`) |
| 노이즈 | 셀마다 `random_uniform_terrain` 추가 (`:167-169`) |

### 7.2 set_terrain 계약 (LLM 생성물)
```python
set_terrain(length, width, field_resolution, difficulty) -> (height_field[m], goals[8,2])
```
- `height_field / vertical_scale → int16`, goals는 셀 좌표
- 지원 시그니처 3종: `(terrain,variation,difficulty)`, `(terrain,difficulty)`, 위 형태 (`terrain_gpt.py:32-46`)
- `fix_terrain` (`:197-306`): 높이 클램프, 골 위치 보정, 스폰 구역 평탄화 — **로그 문자열 "Automatically fixed terrain {id}: ..." 유지 필수**
- `check_terrain_feasibility` (`:355-365`): 골 간 Bresenham 경로 높이 검사
- trimesh 변환: `convert_heightfield_to_trimesh` (grid 방식, `x_edge_mask` 산출 — feet_edge 보상에 필요), pydelatin/pyfqmr 옵션
- Isaac Lab 3.0 주입점: `isaaclab.terrains.utils.create_prim_from_mesh` (확인됨: `terrains/utils.py:63`)

### 7.3 특수 terrain_type (전부 구현 필요)
`default`(set_terrain.py) / `benchmark`(20태스크) / `original` / `original_distill` / `simple` / `random` / 그 외 문자열 → `utils/set_terrains/set_terrain_{type}.py` importlib 동적 로드

## 8. 도메인 랜덤화 (`legged_robot_config.py:299-333`)

| 항목 | 범위 |
|---|---|
| 마찰 | [0.6, 2.0], 64 버킷 |
| base 질량 | +[0, 3] kg |
| base COM | ±0.2 m |
| 밀치기 | 8초마다, 최대 0.5 m/s |
| 모터 강도 | Kp/Kd × [0.8, 1.2] |
| action delay | `action_delay_steps=[1.5]` (증류 시 `--action_delay`) |

## 9. Go1 → Go2 값 매핑 (교차 출처: Extreme-Parkour-Onboard `go2_parkour_config.py` 1차, CAI23sbP 2차)

| 항목 | Go1 (원본) | Go2 (적용값) |
|---|---|---|
| 에셋 | go1_with_camera.urdf | `isaaclab_assets.UNITREE_GO2_CFG` (3.0에 내장 확인) |
| init z | 0.42 | Onboard 값 확인 후 확정 (~0.40-0.42) |
| 기본 관절각 | hip ±0.1 / thigh 0.8(앞),1.0(뒤) / calf −1.5 | 동일 규칙, Onboard 값 대조 |
| Kp / Kd | 20 / 0.5 | **Onboard 값 채택 후 동결** (배포 일치 필수) |
| 토크 한계 | URDF | hip/thigh 23.7, calf 45.4 N·m |
| base_height_target | 0.25 | ~0.30-0.34 (Onboard 대조) |
| 관절명 | FL/FR/RL/RR_{hip,thigh,calf}_joint | 동일 규칙 (USD 프림 대조 필요) |
| 접촉 바디 | 종료: base / 페널티: thigh,calf / 발: *_foot | 동일, USD 이름 대조 |
| 카메라 외부 파라미터 | [0.27, 0.0075, 0.09], pitch 0.52rad | Onboard D435i 마운트 변환 채택 |

⚠️ Isaac Lab 관절 순서는 타입 그룹(모든 hip→thigh→calf), unitree SDK2는 다리별(FR부터) — **`robot.joint_names` 순서를 구현 즉시 obs_spec에 기록**, 배포 시 리맵.

## 10. 학습 스택 (커스텀 rsl_rl — `extreme-parkour/rsl_rl/`, Isaac Gym 의존성 없음 → 재사용)

- `ActorCriticRMA`: scan encoder [128,64,32] / priv encoder [64,20] / actor·critic [512,256,128] / StateHistoryEncoder(conv, tsteps 10)
- `Estimator`: proprio(48) → lin_vel(priv 9 중 3), hidden [128,64], MSE
- DAgger식 priv-reg 스케줄 `[0, 0.1, 2000, 3000]` (resume 시 `_resume` 변형), `dagger_update_freq=20`
- `RecurrentDepthBackbone`: CNN(58×87 입력 계열)+GRU512 → 32 latent (+yaw 예측)
- PPO 추가 옵티마이저 4개 (hist_encoder, estimator, depth_encoder, depth_actor)
- Runner: `learn_RL` / `learn_vision` 분기, 체크포인트 `logs/parkour/{exptid}/model_*.pt`
- Go1 PPO override: `entropy_coef=0.01`

## 11. 증류 (depth student)

| 항목 | 값 |
|---|---|
| 카메라 | 106×60 렌더 → 8px 크롭 → 90×60, FOV 87°, far_clip 2m |
| 갱신 주기 | 5 스텝 (10Hz) — 배포와 일치 필수 |
| depth 버퍼 | 길이 2, `infos["depth"]`로 노출 |
| 학습 | teacher(frozen) 행동 회귀, 192 envs, 10×20 그리드, `num_steps_per_env=120` |
| 트리거 | `--use_camera --action_delay` (distill_eurekaverse.py가 부여) |

## 12. LLM 루프 계약 (백엔드가 반드시 준수해야 하는 것)

### 12.1 CLI (train.py)
`--task {go2} --exptid --device --max_iterations --terrain_type --resume --load_run --checkpoint --check_terrain_feasibility --use_wandb --wandb_id --wandb_group --render_images --use_camera --action_delay --num_envs --proj_name --seed`

### 12.2 CLI (evaluate.py)
`--task --exptid --device --headless --max_steps --metric_granularity {type,level,cell,all} --terrain_type --checkpoint`

### 12.3 로그 문자열 (정확히 일치해야 함)
- train 성공: `"Starting training, using log directory {log_dir}..."` (regex 캡처됨)
- eval 성공: `"Loading model"` (rsl_rl 러너가 출력)
- feasibility 성공: `"Converting heightmap to trimesh"`
- 실패: `"Traceback"`
- 지형 수정: `"Automatically fixed terrain {id}: ..."`

### 12.4 eval stdout 포맷 (`--metric_granularity type`)
```
STATISTICS SUMMARY
Reward: <f>
Reward term <name>: <f>   (반복)
Episode length: <f>
Number of goals reached: <f>
Edge violation: <f>
<빈 줄>
STATISTICS FOR TERRAIN TYPE 00
...동일 키...
<빈 줄>
```
파서: `eurekaverse/utils/terrain_utils.py:83-105`. 키 3개(`Number of goals reached`, `Episode length`, `Edge violation`)와 블록 구분(`\n\n`), 제로패딩 `NN`이 load-bearing.

### 12.5 지형 파일 핸드오프
루프가 `legged_gym/utils/set_terrains/set_terrain_{terrain_type}.py` 작성 → 백엔드 importlib 로드. 신규 구현에서도 동일 경로 규칙 유지 (경로 상수는 `eurekaverse/utils/terrain_utils.py:23`과 맞출 것).

### 12.6 체크포인트 규약
`logs/parkour/{exptid}/model_*.pt`, `--load_run`으로 resume, 시드 체크포인트 exptid `walk_pretrain` (1000 iter, simple 지형).

### 12.7 eurekaverse/utils/terrain_utils.py의 직접 import (재지정 필요)
`isaacgym.terrain_utils`(SubTerrain), `LeggedRobotCfg`, `set_seed`, `terrain_gpt.{fix_terrain, calc_direct_path_heights}` → 신규 백엔드 모듈로 대체 구현.

## 13. 벤치마크 (논문 §5.2)

- `utils/set_terrain_benchmark.py`: 20태스크(+디스패처) × 10난이도, 순수 numpy — 신규 지형 파이프라인으로 그대로 이식
- 판정 기준(원 플랜 M4): (a) Eurekaverse teacher ≥ 자체 학습 original 대조군, (b) iteration 0→4 testing goals 단조 상승, (c) 논문 수치 대비 상대 ~15-20% 이내

## 14. 알고리즘 스케줄 (논문 §4, §5.1 + config)

- walk pretrain 1000 iter → 5 iterations × (지형 생성 → 8 병렬 런 × 2000 iter 학습 → pre/post/all/testing 평가 → best 선택 → 진화)
- 단일 GPU 조정: `max_concurrent_runs=1` 직렬화 패치, `num_parallel_runs=4`, `num_parallel_checks=4` (원 플랜 M5)
- LLM: gpt-5.4 — `n>1` 미지원 → fan-out 패치, pricing dict `.get`, `reasoning_effort="low"` (원 플랜 M3)

## 15. 스택 버전 (신규)

| 구성 | 버전 |
|---|---|
| python | 3.12 (conda `eureka_lab6`) |
| torch | 2.10.0+cu128 |
| Isaac Sim | 6.0.0.1 |
| Isaac Lab | v3.0.0-beta2.patch1 (`~/workspace/IsaacLab3`) |
| 드라이버 | 595.71.05 (Isaac Sim 6.0 최소 595.58 충족) |
| GPU | RTX 3090 24GB — **공식 미지원(최소 RTX 4080), 렌더링 게이트로 실증 필요** |
| 물리 | PhysX (기본값; Newton 백엔드 사용 안 함) |
