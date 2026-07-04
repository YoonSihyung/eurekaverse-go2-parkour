# 코드 완전 해설서 — 혼자서 다루기 위한 가이드

[OPERATING.md](OPERATING.md)가 "어느 버튼을 누르는가"라면, 이 문서는 "기계가 어떻게 돌아가는가"입니다.
모든 경로는 `extreme-parkour/legged_gym/legged_gym/` 기준. 줄 번호는 대략적 위치이니 함수명으로 검색하세요.

---

## 0. 큰 그림: 무엇이 무엇을 부르는가

```
scripts/train.py                          ← 진입점 (CLI 파싱, wandb, 저장)
  └─ utils/task_registry.py make_env()    ← "go2" → (환경 클래스, 설정) 매핑
       └─ envs/base/legged_robot.py       ← ★ 환경 본체 (LeggedRobot)
            ├─ utils/terrain_gpt.py       ← 지형 생성 (set_terrain → heightfield → trimesh)
            └─ envs/base/legged_robot_config.py  ← ★ 모든 파라미터
  └─ task_registry.make_alg_runner()
       └─ rsl_rl/runners/on_policy_runner.py  ← PPO 학습 루프
            ├─ rsl_rl/modules/actor_critic.py ← 정책 신경망 (ActorCriticRMA)
            ├─ rsl_rl/modules/estimator.py    ← 속도 추정기
            └─ rsl_rl/algorithms/ppo.py       ← PPO 갱신 수식

eurekaverse/run_eurekaverse.py            ← LLM 루프 (train.py를 서브프로세스로 호출)
```

**핵심 설계**: 환경(LeggedRobot)과 학습기(rsl_rl)는 서로를 모릅니다. 러너가 `env.step(actions)`을 호출하고 `(관측, 보상, 종료, 타임아웃, 정보)`를 받는 것이 유일한 접점입니다.

## 1. 한 스텝의 해부 — `LeggedRobot.step()` (legged_robot.py ~150행)

정책이 액션을 내면 어떤 일이 일어나는지, 순서 그대로:

```
step(actions):                                  # 50Hz (0.02초)
  1. _pre_physics_step(actions)                 # 액션 클리핑, 히스토리 버퍼 기록
  2. for _ in range(decimation=4):              # 물리는 200Hz (0.005초 × 4)
       _apply_action()                          #   목표각 = action×0.25 + 기본자세
       sim.step()                               #   액추에이터(IdealPD)가 토크 계산 → PhysX
  3. post_physics_step():
       roll/pitch/yaw 갱신 (쿼터니언→오일러)     # ← xyzw 규약 주의! (§9 함정)
       _update_goals()                          #   골 도달 판정, target_yaw 계산
       _get_dones()                             #   종료(넘어짐)/타임아웃(20초 or 8골 완주)
       _get_rewards()                           #   14개 보상 항 합산
       _reset_idx(죽은 env들)                   #   리셋 (커리큘럼 승급/강등도 여기서)
       _get_observations()                      #   698차원 관측 조립
  4. return (obs, privileged_obs), reward, reset_term, reset_time_out, extras
```

**수정할 때의 감각**: "로봇이 매 스텝 무엇을 아는가" = 3의 관측 조립, "무엇을 원하도록 만들 것인가" = 3의 보상, "몸이 어떻게 반응하는가" = 2의 액추에이터/config.

## 2. 관측 698차원 — `_get_observations()` (legged_robot.py ~500행)

```
[ proprio 48 | scandots 132 | priv_explicit 9 | priv_latent 29 | history 480 ]
```

| 블록 | 내용 | 실기에서 얻을 수 있나? |
|---|---|---|
| proprio 48 | 각속도3, roll/pitch2, 골방향각2, 속도명령1, 관절각12, 관절속도12, 직전액션12, 발접촉4 | ✅ (배포 시 IMU/엔코더로 재구성) |
| scandots 132 | 로봇 주변 12×11 격자의 지형 높이 | ❌ 특권정보 → 증류로 depth 카메라 대체 |
| priv_explicit 9 | 몸통 선속도3 (+패딩6) | ❌ → Estimator가 proprio에서 추정 |
| priv_latent 29 | 질량4, 마찰1, 모터게인 계수24 | ❌ → 히스토리 인코더가 암묵 추정 (RMA) |
| history 480 | 최근 10스텝의 proprio (yaw 채널은 0 마스킹) | ✅ |

**채널을 추가/삭제하려면**: ① proprio에 넣을 거면 `n_proprio` (config `CustomEnvCfg`)를 바꾸고 `_get_observations`의 `torch.cat` 순서에 삽입 ② 신경망이 `n_proprio` 등으로 입력을 슬라이스하므로 **config 숫자와 cat 순서가 일치해야 함** (assert가 잡아줌) ③ 배포까지 갈 채널이면 실기에서 재구성 가능한지 먼저 자문할 것.

## 3. 보상 시스템 — 자동 등록 마법

`_prepare_reward_function()` (legged_robot.py ~870행)이 하는 일:
```python
for 이름, 스케일 in cfg.rewards.scales:   # RewardScales 클래스의 모든 속성
    if 스케일 != 0:
        self._reward_함수목록에 self._reward_<이름> 등록
        유효스케일 = 스케일 × dt(0.02)     # ← 스케일에 dt가 곱해짐!
```
매 스텝: `총보상 = clip(Σ 항별보상×유효스케일, min=0)` (`only_positive_rewards=True`).

**이해 포인트**:
- 스케일 −10이 커 보여도 실제론 −0.2/스텝. 양성 항 최대치는 tracking_goal_vel 0.03 + tracking_yaw 0.01/스텝 → **페널티와 양성의 균형이 매우 민감**합니다. 스케일 조정은 한 번에 하나씩, 2~5배 단위로.
- `only_positive_rewards` 때문에 페널티가 양성을 넘으면 그 스텝 보상은 0으로 잘림 → "아무것도 안 하기"가 국소최적이 되기 쉬움. 우리가 겪은 기는 문제의 배경 메커니즘.
- 새 보상 추가: `RewardScales`에 `내보상 = 0.5` + env에 `def _reward_내보상(self): return <shape (num_envs,) 텐서>` — 끝.
- 조건부 보상 패턴: `rew[self.env_class != -1] = 0` (평지에서만) 같은 마스킹을 원본이 즐겨 씀.

## 4. 지형 시스템 — 이 프로젝트의 특수 장기

### 4.1 계약: `set_terrain` 함수
LLM이 생성하는 것도, 인간이 짜는 것도 같은 형태입니다:
```python
def set_terrain(length, width, field_resolution, difficulty):  # 미터 단위, difficulty 0~1
    height_field = np.zeros((length/res, width/res))   # 높이맵 (미터)
    goals = np.zeros((8, 2))                           # 골 8개 (셀 내 좌표)
    ...  # numpy로 자유롭게 지형 조각
    return height_field, goals
```
`utils/set_terrain_simple.py`를 열어보면 15줄짜리 최소 예제가 있습니다. **커스텀 지형 = 이 파일 복사 → 수정 → `--terrain_type 내지형` (파일명 `set_terrain_내지형.py`를 `utils/set_terrains/`에)**.

### 4.2 그리드와 커리큘럼
- 전체 지형 = 10행(난이도 0→1) × 40열(변형) 격자, 셀당 18×4m
- 로봇마다 (행, 열)이 배정되고, **골 도달률로 승급/강등**: 8골 중 ≥80% → 어려운 행으로, <40% → 쉬운 행으로 (`_update_terrain_curriculum`)
- 셀별 시드 `int(variation*1e3 + difficulty*1e6)` → 같은 셀은 항상 같은 지형 (재현성)

### 4.3 파이프라인
`terrain_gpt.py`: heightfield → `fix_terrain`(GPT 실수 자동수정: 높이 클램프, 골 위치 보정, 스폰 평탄화) → trimesh 변환(+`x_edge_mask` — feet_edge 보상용 절벽 마스크) → **300만 삼각형 초과 시 청크 분할**(PhysX 한계, 우리가 추가) → USD 주입.

## 5. 학습 스택 — rsl_rl의 커스텀 구조

```
관측 698 ─┬─ proprio 48 ──────────────┬──→ Actor MLP [512,256,128] → 액션 12
          ├─ scandots 132 → ScanEncoder ┤        ↑
          ├─ priv_explicit 9 ───────────┤   (추론 시 Estimator 출력으로 대체)
          ├─ priv_latent 29 → PrivEncoder┤   (추론 시 HistoryEncoder 출력으로 대체)
          └─ history 480 → HistoryEncoder┘
```
- **왜 이런 구조?** 학습 때는 특권정보(진짜 속도, 진짜 마찰)로 편하게 배우고, 배포 때는 그것들을 proprio/히스토리에서 **추정**해서 대체 (RMA/teacher-student). Estimator는 MSE로, HistoryEncoder는 PrivEncoder 출력을 따라하도록(DAgger, `dagger_update_freq=20`) 함께 훈련됩니다.
- PPO 하이퍼파라미터: config `LeggedRobotCfgPPO` — lr 2e-4(adaptive KL), γ 0.99, λ 0.95, entropy 0.01, 24 스텝/env/iter
- 체크포인트(`model_*.pt`)에는 actor_critic + estimator + 옵티마이저 상태가 함께 저장 → `--resume --load_run 이름`으로 이어짐

## 6. LLM 루프 (eurekaverse/) — 아직 연결 작업 중

`run_eurekaverse.py`가 하는 일 (iteration마다): GPT에게 지형 코드 생성 요청 → feasibility 체크(train.py --check_terrain_feasibility 서브프로세스) → 통과한 지형으로 N개 병렬 학습(train.py 서브프로세스) → 평가(evaluate.py) → **골 도달 수로 최고 정책 선정** → 그 지형+성적을 GPT에게 피드백으로 주고 진화 요청 → 반복. 백엔드와는 CLI/로그 문자열/파일로만 통신 (그래서 우리가 백엔드를 통째로 갈아도 루프가 사는 것).

## 7. 실전 수정 레시피 모음

| 하고 싶은 것 | 방법 |
|---|---|
| 더 빨리 뛰게 | `CmdRanges.lin_vel_x = [0.5, 1.2]` (단, tracking_goal_vel이 명령 속도로 정규화되므로 보상 균형 확인) |
| 특정 행동 억제 (예: 점프 남발) | `RewardScales.lin_vel_z`를 −1→−2로 (평지 외에도 적용하려면 `_reward_lin_vel_z`의 마스킹 제거) |
| 지형 난이도 커브 조절 | 각 set_terrain 함수 안의 `difficulty` 사용부 (예: `gap_width = 0.75 + 0.25*difficulty`) |
| 커리큘럼 더 빡세게 | `_update_terrain_curriculum`의 0.8/0.4 임계값 |
| 에피소드 길이 | `episode_length_s = 20` |
| 새 로봇 | `UNITREE_GO2_CFG` 복사 → usd 경로/게인/자세 수정 → `Go2Cfg` 상속 클래스 → `envs/__init__.py`에 등록 |
| DR 강도 | `CustomDomainRandCfg`의 range들 (예: friction_range [0.6,2.0]) |
| 학습 속도/안정 트레이드오프 | num_envs(샘플량), learning_rate, num_steps_per_env |

## 8. 검증 루틴 — 이 프로젝트의 교훈이 응축된 부분

수정 후 **반드시 이 순서로**:
```bash
# 1) 짧은 학습이 도는가 (5분)
python -u scripts/train.py --task go2 --exptid test1 --max_iterations 30 \
    --terrain_type simple --num_envs 2048 --headless

# 2) 결정적 지표 — 골 도달 (보상 상승만 믿지 말 것!)
python -u scripts/evaluate.py --task go2 --exptid test1 --headless \
    --max_steps 1001 --metric_granularity type --terrain_type simple
# → "Number of goals reached"가 유일하게 거짓말 안 하는 지표

# 3) 눈으로 (../../../view.sh test1)
```
**교훈**: 이 프로젝트에서 "보상이 오르고 에피소드가 길어지는" 학습이 사실은 기어다니고 있었습니다. 보상은 속아도 골 도달은 안 속습니다.

계측 진단 도구 (scripts/에 있음, 필요시 복사·수정):
- `diag_spawn.py` — 제로 액션으로 스폰 상태·종료 원인 관찰
- `diag_contacts.py` — 센서별 접촉력 (보상 오염 검사)
- `diag_goalsig.py` — 골 방향 관측이 기하학적으로 맞는지
- `diag_view.py` — 뷰포트 스크린샷 캡처

## 9. 함정 목록 (우리가 밟은 지뢰들)

1. **쿼터니언 규약**: Isaac Lab 3.0은 (x,y,z,w). 쿼터니언을 수동으로 인덱싱하는 코드를 쓰면 안 되고, 반드시 `isaaclab.utils.math` 함수를 쓰거나 w가 마지막임을 기억할 것. 서 있는 로봇의 quat가 [0,0,0,±1]이면 xyzw입니다.
2. **PhysX 충돌 메시 한계**: 단일 메시 ~300만 삼각형 초과 시 쿠킹이 **조용히** 실패 → 로봇이 지형을 통과. 로그에서 `Too many child nodes` 검색. (우리 임포터는 자동 청크 분할함)
3. **평가와 학습의 설정이 다름**: evaluate.py는 학습 때 저장된 pkl 설정을 로드하고 일부(커리큘럼, 일부 DR)를 끕니다. "학습에서 잘 됐는데 평가가 이상"하면 evaluate.py 상단의 설정 오버라이드부터 볼 것.
4. **reset_term vs reset_time_out**: 타임아웃만 가치 부트스트랩됩니다. 이 구분을 건드리면 학습이 미묘하게 망가짐 (부호가 아니라 분포가 틀어져서 찾기 어려움).
5. **`self.dt`(0.02, 정책)와 `sim.dt`(0.005, 물리)**: 보상 스케일·타이머는 전부 정책 dt 기준.
6. **GUI**: 학습은 headless가 기본. 보려면 `--gui --viz kit` + GLib LD_PRELOAD (view.sh/train_view.sh가 처리).
7. **원본 대조가 필요한 수정이면**: `~/workspace/eurekaverse`(원본)와 diff하고, [verification_report.md](verification_report.md)에 항목 추가 — 이 문서가 "이 값이 왜 이 값인지"의 원장입니다.

## 10. 스스로 학습하는 코스 (권장 순서)

1. `set_terrain_simple.py`를 복사해 계단 지형을 만들어 `view.sh`로 확인 (지형 계약 체득)
2. `RewardScales.tracking_yaw`를 0으로 놓고 30 iter 학습 → 로봇이 방향을 못 잡는 것 관찰 (보상 항의 역할 체득)
3. `diag_goalsig.py`를 읽고 관측 채널 하나를 직접 출력해보기 (관측 파이프라인 체득)
4. Kp를 40→20으로 낮춰 300 iter → 걸음 품질 비교 (액추에이터 감각)
5. `_reward_` 함수 하나를 처음부터 작성 (예: 발 높이 클리어런스 보상)

이 다섯 개를 해보면 이 코드베이스의 90%를 다룰 수 있습니다.
