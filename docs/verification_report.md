# 원본 대조 검증 리포트

방법: 원본(`~/workspace/eurekaverse`, Isaac Gym)과 본 구현의 보상·상수·로직을 스크립트 추출 + 라인 단위 대조.
기준 문서: [reproduction_spec.md](reproduction_spec.md). 상태: **진행 중** (완료 항목만 아래 기록).

## 1. 보상 스케일 (14항) — 검증 완료

| 항목 | 원본 | 포팅 베이스 | 판정 | 조치 |
|---|---|---|---|---|
| tracking_goal_vel | 1.5 | 1.5 | ✅ | |
| tracking_yaw | 0.5 | 0.5 | ✅ | |
| lin_vel_z | −1.0 | −1.0 | ✅ | |
| ang_vel_xy | −0.05 | −0.05 | ✅ | |
| orientation | −1.0 | −1.0 | ✅ | |
| **dof_acc** | **−2.5e-7** | −8.25e-8 | ❌ 편차 | **−2.5e-7 복원** |
| collision | −10.0 | −10.0 | ✅ | (함수 본문은 아래 참조) |
| action_rate | −0.1 | −0.1 | ✅ | |
| delta_torques | −1e-7 | −1e-7 | ✅ | |
| torques | −1e-5 | −1e-5 | ✅ | |
| hip_pos | −0.5 | −0.5 | ✅ | |
| dof_error | −0.04 | −0.04 | ✅ | |
| feet_stumble | −1 | −1 | ✅ | |
| feet_edge | −1 | −1 | ✅ | |

## 2. 보상 함수 본문 (14개) — 검증 완료, 편차 3건 교정

| 함수 | 판정 | 상세 |
|---|---|---|
| **_reward_collision** | ❌→✅ 교정 | 포팅 베이스가 `1.*(...)`을 `0.*(...)`으로 바꿔 **페널티를 무력화**해놨음 (디버깅 잔재). 원본 복원 (legged_robot.py) |
| **_reward_tracking_goal_vel** | ❌→✅ 교정 | 원본은 **월드 프레임** 속도(`root_states[:,7:9]`)를 월드 프레임 골 방향과 내적. 포팅 베이스는 바디 프레임(`root_lin_vel_b`) 사용 → `root_lin_vel_w`로 교정 |
| **_reward_dof_acc** | ❌→✅ 교정 | 원본은 정책 스텝(0.02s) 유한차분 `((last_dof_vel−dof_vel)/dt)²`. 포팅 베이스는 솔버 `joint_acc²`(물리 dt 기반, 값 스케일 상이). 원본 계산식 복원 (`last_dof_vel` 버퍼는 이미 유지되고 있었고 갱신 타이밍도 원본과 동일함을 확인) |
| _reward_feet_edge | ✅ | 본문+꼬리(contact_filt AND, terrain_level>3 게이팅) 완전 일치 |
| _reward_action_rate | ✅ | `_previous_actions` 갱신이 보상 계산 후(원본과 동일 타이밍) |
| _reward_feet_stumble | ✅ | 접촉력 xy>4·z 판정 동일 (양쪽 다 월드 프레임) |
| 나머지 8개 | ✅ | API 이름 변경만 (`base_ang_vel`→`root_ang_vel_b` 등), 수식 동일 |

`only_positive_rewards=True`, `tracking_sigma=0.2`, 보상 스케일 dt 곱(레거시 `_prepare_reward_function` 방식) 일치 확인.

## 3. 상수 전수 대조 — 편차 4건 교정

| 항목 | 원본 | 포팅 베이스 | 조치 |
|---|---|---|---|
| max_init_terrain_level | 2 | 1 | **2 복원** |
| randomize_base_com | True (±0.2m) | False | **True 복원** (구현은 존재, 플래그만 꺼져 있었음) |
| terrain num_rows × num_cols | 10 × 40 | 8 × 20 | **10×40 복원** (RTX 3090 메모리 실측 예정; 부족 시 CLI로 축소하고 편차 기록) |
| horizontal_scale | 0.05 | 0.1 | **0.05 복원** (0.1은 원본 주석상 depth 학습 시에만 권장) |

일치 확인된 것: next_goal_threshold 0.2 / reach_goal_delay 0.1 / episode_length_s 20 / clip_actions 1.2 / clip_observations 100 / obs_scales(lin_vel 2.0, ang_vel 0.25, dof_pos 1.0, dof_vel 0.05, height 5.0) / friction_range [0.6,2.0] / added_mass_range [0,3] / push_interval 8s·0.5m/s / motor_strength_range [0.8,1.2] / entropy_coef 0.01 / PPO 하이퍼파라미터 / soft_dof_pos_limit 0.9 / base_height_target 0.25 (base+task 구조 차이일 뿐 유효값 동일).

obs noise: 원본도 `add_noise=False`(미사용) → 포팅 베이스에서 제거된 것은 편차 아님.

## 3.5 도메인 랜덤화 — 중대 편차 발견 및 복원 구현

**발견**: 포팅 베이스는 DR 4종이 전부 미구현이었음 (`_init_robot`에서 플래그가 켜져 있으면 `raise NotImplementedError`; 포크는 4개 플래그를 모두 False로 꺼두고 학습). 그 결과:
- 마찰/질량/COM/모터 랜덤화가 동역학에 전혀 반영되지 않음
- 특권 관측 `mass_params_tensor`(4dim)=0, `friction_coeffs_tensor`=1 상수 → **RMA priv_latent 학습 무의미화**
- 논문의 sim2real 전제(DR로 훈련된 estimator/adaptation) 훼손

**복원 구현** (`legged_robot.py: _apply_domain_randomization`, init 시 1회 — 원본과 동일):

| 항목 | 원본 의미론 | 복원 구현 (Isaac Lab 3.0 API) |
|---|---|---|
| 마찰 | 64버킷 U[0.6,2.0], env당 단일 계수, 모든 shape 적용 | `root_view.get/set_material_properties` (static+dynamic 동일값) |
| base 질량 | += U[0,3]kg, **recomputeInertia=True** | `set_masses_index` + 질량비로 `set_inertias_index` |
| base COM | += U[−0.2,0.2] xyz | `set_coms_index` (body_com_pose_b 기반) |
| 모터 강도 | Kp/Kd × U[0.8,1.2] (2,N,12) — 토크 계산에 적용 | actuator `stiffness/damping` 텐서에 per-env 곱 (PD형 액추에이터에서 유효) |
| motor_strength 샘플링 | 플래그 off 시 1.0 (priv obs=0) | 조건부 샘플링으로 교정 (베이스는 무조건 랜덤 샘플 → obs 오염이었음) |
| mass_params 관측 | [added_mass, com_xyz] | 동일 레이아웃 유지 |

config 플래그 4종 모두 원본값(True) 복원.

## 4. 알려진 잔여 편차 (작업 예정)

| 항목 | 원본 | 현재 | 계획 |
|---|---|---|---|
| **제어 경로** | 명시적 PD (Kp20/Kd0.5) + 토크 클리핑 | position target → ActuatorNetMLP(Go1 학습 모터 모델) | Go2 전환 시 **DCMotorCfg(명시적 PD + 모터 곡선 클램프)**로 교정. Kp40/Kd1 (Onboard 실기 검증값), effort 23.5/45.4 |
| **motor_strength DR** | Kp/Kd에 ×[0.8,1.2] 적용 (동역학 반영) | priv_latent 관측에만 사용 | Go2 env에서 per-env actuator stiffness/damping에 랜덤 계수 적용 복원 |
| dof_acc의 last_dof_vel 소스 | 자체 계산 토크 기반 dof_vel | Isaac Lab joint_vel | 동등 (상태 소스만 다름) |
| 물리 엔진 | PhysX(Isaac Gym Preview) | PhysX 5 (Isaac Sim 6.0) | 불가피한 플랫폼 차이 — 벤치마크로 거동 비교 |

## 5. 2차 감사 (로직 블록) — 완료

| 블록 | 판정 | 상세 |
|---|---|---|
| 관측 조립 (`_get_observations`) | ✅ | proprio 순서/스케일/프레임, priv_explicit(바디 프레임 — 원본 동일), priv_latent, scandots(z−0.3), yaw 히스토리 마스킹 전부 일치 |
| `contact_buf` | 무해 편차 | 포팅본에 없음 — 원본에서도 소비처가 없는 죽은 버퍼라 생략 유지 |
| 골 로직 (`_update_goals`) | ✅ | 완전 일치 |
| **종료 분류** | ❌→✅ 교정 | 원본은 8골 완주를 **타임아웃**(가치 부트스트랩 O)으로 분류. 포팅본은 termination(부트스트랩 X)으로 분류 → 완주 페널티화. 타임아웃으로 교정 |
| 커리큘럼 (goal 기반) | ✅ | 승급 0.8/강등 0.4/정체 25% 랜덤화 일치 |
| 커리큘럼 (original 지형용 거리 기반) | ⚠️ TODO | 포팅본 `NotImplementedError` — **M9 대조군(`--terrain_type original`) 학습 전 구현 필요** |
| 커맨드 리샘플 | ✅ | 완전 일치 |

## 6. 인프라 이슈 (해결됨)

- **PhysX 5 쿠킹 한계**: horizontal_scale 0.05 × 10×40 그리드 = ~2,600만 삼각형 단일 메시 → BV4 쿠킹 실패 → 충돌 무효(로봇 낙하 즉사). **지형 메시를 300만 삼각형 이하 청크로 x축 분할 임포트**로 해결 (기하 변형 없음, `terrain_gpt.py: TrimeshTerrainImporter.import_mesh`). 충돌 필터 global prim 목록도 청크 경로로 갱신.

## 7. Go2 구성 (작성 완료)

- `UNITREE_GO2_CFG`: go2.usd + **명시적 PD(IdealPDActuator)** Kp40/Kd1(실기 검증값) — 원본 제어 의미론 복원, ActuatorNet 편차 해소
- ⚠️ 교훈 기록: 최초에 DCMotorCfg로 구성했으나 **DCMotor의 속도-토크 커브(속도 비례 토크 감쇠)는 원본에 없는 추가 제약**으로 판명 — 1000 iter walk_pretrain이 기어다니는 국소최적에 빠짐(골 도달 0, collision −52). 원본 `_compute_torques`(PD + 정적 토크 클립)와 정확히 일치하는 IdealPDActuator로 교체 후 재학습.
- 모터 스펙 2그룹 분리: hip/thigh 23.7N·m@30.1rad/s, calf 45.43N·m@15.7rad/s (단일값 구현들의 calf 과소설정 교정)
- self-collision 비활성 (원본값; 포크 Go1은 활성 — 편차로 기록)
- motor_strength DR이 PD 게인에 작용 → 원본 `_compute_torques` 의미론 완성

## 검증 이력

- 2026-07-03: 1차 감사 (보상 스케일 14+함수 14+상수 전수) — 편차 7건 교정
- 2026-07-03: DR 4종 미구현 발견 → 복원 구현
- 2026-07-03: PhysX 쿠킹 이슈 진단·해결 (청크 분할)
- 2026-07-03: 2차 감사 (관측/골/종료/커리큘럼/커맨드) — 종료 분류 1건 교정, original 커리큘럼 TODO
- 2026-07-03: Go2 구성 작성, go1/go2 스모크 통과 (30 iter 학습 성장 확인)

## 남은 감사 항목

- terrain_gpt.py `fix_terrain`/`check_terrain_feasibility` 원본 대비 (루프 연결 전)
- evaluate.py 통계 산출 경로 (STATISTICS 블록 생성부)
- rsl_rl 2파일 diff (estimator.py, on_policy_runner.py — 포크 어댑테이션 검토)

## 8. 3차 감사 — "못 걷는" 근본 원인 (2026-07-05)

프로브 매트릭스(R1 go1 / R2 DR-off / R3 collision-off 전부 골 0)로 Go2·DR·collision을 배제한 뒤 계측 진단으로 확정:

| # | 편차/버그 | 증상 | 교정 |
|---|---|---|---|
| 13 | **쿼터니언 규약**: Isaac Lab 3.0이 (w,x,y,z)→**(x,y,z,w)**로 변경. 포팅 베이스의 `euler_from_quaternion`은 w-first 가정 | roll/pitch/yaw 전부 쓰레기 → delta_yaw(골 방향 관측)·IMU 관측·종료 판정 오염 → **목적 있는 보행 학습 불가능** (기어다니는 국소최적) | 인덱싱 교정 (`legged_robot.py:71`), 계측 재검증 완료 (yaw 0.02, delta_yaw=기하학값 일치). 다른 쿼터니언 소비처는 전부 3.0 자체 math 함수라 일관성 확인 |
| 14 | **커맨드 런타임 오버라이드**: 포크 helpers.py가 terrain_type별로 lin_vel_x를 [0,2.0](simple)/[0.3,1.2](else)로 강제 — 원본에는 없는 로직 | 걷기 사전학습에 0~2m/s 혼합 커맨드(정지 포함+전력질주) 부과 | 오버라이드 제거 → 원본 활성값 [0.3,0.8] 사용 |

교훈: 포크의 "학습이 되는 것처럼 보였던" 이전 실행들(보상 상승, 에피소드 생존)은 전부 이 상태에서의 결과였음 — 골 도달 지표를 통한 검증이 필수.
