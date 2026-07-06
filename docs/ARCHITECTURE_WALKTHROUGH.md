# 프로젝트 구조 · Isaac Lab 세팅 · 논문 구현 해설

[CODE_GUIDE.md](CODE_GUIDE.md)가 "기계가 어떻게 돌아가는가"라면, 이 문서는 그보다 한 단계 앞선 **"무엇이 무엇이고, 어디를 보면 되는가"**를 정리한 것입니다. 세 부분으로 나뉩니다.

1. [프로젝트 전체 구조와 파일 역할](#1-프로젝트-전체-구조와-파일-역할)
2. [Isaac Sim / Isaac Lab 자체 세팅 코드](#2-isaac-sim--isaac-lab-자체-세팅-코드)
3. [논문 내용이 어떻게 구현됐는가](#3-논문-내용이-어떻게-구현됐는가)

모든 경로는 저장소 루트(`eurekaverse_lab3/`) 기준.

---

## 1. 프로젝트 전체 구조와 파일 역할

이 저장소는 **세 개의 독립적인 층**이 합쳐진 것입니다. 이 구분을 먼저 잡으면 나머지가 쉽게 들어옵니다.

```
eurekaverse_lab3/
├── extreme-parkour/       ← 층 A+B: "시뮬레이터 환경 + RL 학습기" (실제 로봇을 학습시키는 엔진)
│   ├── legged_gym/        ←   층 A: 환경 (Isaac Lab 위에서 로봇/지형/보상 정의)
│   └── rsl_rl/            ←   층 B: 학습기 (PPO + teacher-student 신경망)
├── eurekaverse/           ← 층 C: "LLM 진화 루프" (지형을 GPT로 생성·진화, A+B를 서브프로세스로 부림)
└── docs/                  ← 문서
```

**핵심 설계 원칙** (CODE_GUIDE §0에도 나오는): 이 세 층은 서로를 직접 import하지 않습니다.

- 환경(A)과 학습기(B)의 유일한 접점은 `env.step(actions) → (obs, reward, done, ...)`.
- LLM 루프(C)는 A+B를 **CLI 서브프로세스 + 로그 문자열 파싱**으로만 조종합니다 (그래서 백엔드를 통째로 갈아치워도 루프가 삽니다).

### 층 A — 환경: `extreme-parkour/legged_gym/legged_gym/`

| 파일 | 역할 |
|---|---|
| `envs/base/legged_robot.py` | ★ **환경 본체** (1329줄). `LeggedRobot` 클래스. 물리 스텝, 관측 조립, 14개 보상, 종료 판정, 리셋, 커리큘럼, 도메인 랜덤화가 전부 여기. |
| `envs/base/legged_robot_config.py` | ★ **모든 파라미터** (633줄). 로봇 articulation(Go1/Go2), 시뮬 설정, 관측 차원, 보상 스케일, PPO 하이퍼파라미터. |
| `envs/__init__.py` | `"go1"`, `"go2"` 태스크를 레지스트리에 등록. |
| `utils/task_registry.py` | 태스크 이름 → (환경 클래스, config) 매핑. `make_env()`, `make_alg_runner()`. |
| `utils/terrain_gpt.py` | **지형 파이프라인** (393줄). heightfield → 자동수정(`fix_terrain`) → trimesh → USD 주입. `Terrain`, `TrimeshTerrainImporter`. |
| `utils/set_terrain_*.py` | **지형 정의 함수들**. 각각 `set_terrain(length,width,res,difficulty)` 계약을 구현. `simple`(최소예제), `original`(논문 원본), `benchmark`(평가용 20태스크), 그리고 LLM이 생성하는 `set_terrain_it-N_run-M.py`. |
| `scripts/train.py` | **학습 진입점**. CLI 파싱 → Isaac Sim 앱 런칭 → env+runner 생성 → `runner.learn()`. |
| `scripts/evaluate.py` | **평가 진입점**. 저장된 정책으로 골 도달 수 측정 (612줄, 이 프로젝트의 "진실 지표"). |
| `scripts/diag_*.py` | 진단 도구 (스폰 상태, 접촉력, 골 신호, 스크린샷). |

### 층 B — 학습기: `extreme-parkour/rsl_rl/rsl_rl/`

| 파일 | 역할 |
|---|---|
| `runners/on_policy_runner.py` | **PPO 학습 루프**. 롤아웃 수집 → 업데이트 반복, 체크포인트 저장. |
| `modules/actor_critic.py` | **정책 신경망** (`ActorCriticRMA`). 관측 인코더들 + Actor/Critic MLP. |
| `modules/estimator.py` | **속도 추정기**. proprio에서 특권정보(몸통 속도) 추정. |
| `modules/depth_backbone.py` | 증류 단계의 **depth 카메라 인코더** (scandots 대체). |
| `algorithms/ppo.py` | **PPO 갱신 수식** + DAgger(히스토리 인코더 증류). |
| `storage/rollout_storage.py` | 롤아웃 버퍼(GAE 계산). |

### 층 C — LLM 진화 루프: `eurekaverse/`

| 파일 | 역할 |
|---|---|
| `run_eurekaverse.py` | ★ **메인 진화 루프**. iteration마다 지형 생성→학습→평가→best 선택→진화. |
| `distill_eurekaverse.py` | teacher(scandots) → student(depth) **증류** 실행. |
| `gpt/system_prompt.txt` 외 | GPT에게 주는 **프롬프트 템플릿**(지형 함수 작성 지침) + 예시 지형 코드. |
| `utils/gpt_utils.py` | GPT API 호출, 응답 파싱. |
| `utils/terrain_utils.py` | eval stdout 파싱(골 도달 수 추출), 지형 파일 핸드오프. |
| `config/config.yaml` | 루프 설정(iteration 수, 병렬 런 수, 학습 iter 수 등). |

### docs

| 파일 | 역할 |
|---|---|
| `reproduction_spec.md` | 원본(Isaac Gym+Go1) 대비 **모든 수치의 ground truth**. "이 값이 왜 이 값인지"의 원장. |
| `CODE_GUIDE.md` | 코드 해설서 (한 스텝의 해부, 보상 시스템, 지형 시스템 등). |
| `OPERATING.md` / `NEXT_STEPS.md` / `verification_report.md` | 운영 매뉴얼 / 인수인계 / 원본 대조 검증 기록. |

**한 줄 요약**: `extreme-parkour`가 진짜 로봇을 학습시키는 엔진이고, `eurekaverse`는 그 엔진에게 "어떤 지형에서 연습시킬지"를 GPT로 자동 설계·진화시키는 메타 루프입니다.

---

## 2. Isaac Sim / Isaac Lab 자체 세팅 코드

이 프로젝트는 원래 **Isaac Gym**(구형)으로 짜인 것을 **Isaac Lab 3.0 + Isaac Sim 6.0**으로 이식한 것입니다. Isaac Lab 관련 코드는 크게 6곳에 있습니다. (코드의 주석 대부분이 "Isaac Gym→Lab 3.0 이식하며 무엇이 바뀌었나"를 기록하고 있어, 이식 이해엔 주석이 최고의 자료입니다.)

### 2.1 앱 런칭 — `scripts/train.py:44-67`

Isaac Sim은 **먼저 시뮬레이터 앱을 띄운 뒤에야** 나머지 isaaclab 모듈을 import할 수 있습니다. 그래서 순서가 중요합니다:

```python
from isaaclab.app import AppLauncher     # 앱 런처만 먼저
...
app_launcher = AppLauncher(args)          # ← 여기서 Isaac Sim 커널이 부팅됨
simulation_app = app_launcher.app
from legged_gym.envs import *             # ← 부팅 후에야 환경/isaaclab 모듈 import
```

- `--headless`가 기본 (`train.py:59-61`) — 학습은 GUI 없이. 보려면 `--gui`.
- `AppLauncher.add_app_launcher_args(parser)`가 `--device`, `--headless` 등 Isaac Lab 표준 인자를 파서에 주입합니다.

### 2.2 환경 베이스 클래스 — `DirectRLEnv` 상속

`envs/base/legged_robot.py:89`:

```python
class LeggedRobot(DirectRLEnv):
```

Isaac Lab에는 두 가지 환경 스타일(Manager-based / Direct)이 있는데, 이 프로젝트는 원본 Isaac Gym 코드를 최대한 그대로 옮기려고 **Direct** 스타일을 씁니다. `DirectRLEnv`가 요구하는 훅들을 구현한 것이 이 클래스입니다:

- `_setup_scene()` — 씬 구성 (지형, 로봇, 조명)
- `_pre_physics_step()` / `_apply_action()` — 물리 스텝 전 처리
- 그리고 원본 로직을 살리기 위해 **`step()`을 직접 오버라이드** (`legged_robot.py:144`)

#### step()에서 Isaac Lab 특유의 부분 (`legged_robot.py:163-181`)

```python
for _ in range(self.cfg.decimation):        # 정책 1스텝 = 물리 4스텝
    self._apply_action()                     # 관절 목표각 세팅
    self.scene.write_data_to_sim()           # ← 여기서 액추에이터가 토크 계산해 PhysX로 write
    self.sim.step(render=False)              # 물리 적분
    if ... and is_rendering:
        self.sim.render()                    # GUI/카메라 필요할 때만 렌더 (성능)
    self.scene.update(dt=self.physics_dt)    # 센서 버퍼 갱신
```

CODE_GUIDE §9의 함정 5번(`self.dt`=0.02 정책 vs `sim.dt`=0.005 물리)이 바로 이 `decimation=4` 루프에서 나옵니다.

### 2.3 시뮬레이션 설정 — `legged_robot_config.py:480-507`

`LeggedRobotCfg(DirectRLEnvCfg)` 안의 `sim: SimulationCfg`:

```python
sim = SimulationCfg(
    device="cuda:0",
    dt=0.005,                                # 물리 200Hz
    gravity=(0.0, 0.0, -9.81),
    physics=PhysxCfg(                        # ← 3.0에서 physx→physics로 이름 바뀜
        solver_type=1,
        max_position_iteration_count=4,
        gpu_max_rigid_contact_count=2**24,
        gpu_max_rigid_patch_count=2**21,     # ← 우리가 키운 값 (아래 설명)
        bounce_threshold_velocity=0.5,
    ),
    render_interval=4
)
```

**여기 주석들이 곧 이식 기록입니다**:

- `PhysxCfg`는 3.0에서 `isaaclab_physx.physics`로 위치가 옮겨짐 (`config.py:6`).
- `gpu_max_rigid_patch_count=2**21`: 6144개 환경이 계단/박스 지형을 밟으면 접촉 패치가 ~100만 개 필요한데, 3.0 기본값(163k)이 넘쳐서 **조용히 접촉을 버립니다**(로봇이 지형 통과). 원본 Isaac Gym은 `default_buffer_size_multiplier=5`로 해결했던 것을 여기선 이 값으로 대응. (CODE_GUIDE §9 함정 2번)

### 2.4 로봇 정의 — Articulation (`config.py:80-139`)

`UNITREE_GO2_CFG = ArticulationCfg(...)`가 로봇 자체를 정의합니다:

- **`spawn=UsdFileCfg(usd_path=".../Go2/go2.usd")`** — Isaac Sim의 Nucleus 자산 서버에서 Go2 USD 모델을 불러옴.
- **`init_state`** — 스폰 높이 0.42m, 각 관절의 기본 각도(action=0일 때 목표각). 이 12개 값이 `default_joint_pos`.
- **`actuators`** — 이식에서 가장 중요한 부분:

```python
"hip_thigh": IdealPDActuatorCfg(
    effort_limit=23.7, velocity_limit=30.1,
    stiffness=40.0,   # Kp
    damping=1.0,      # Kd
)
```

`IdealPD` = "목표각 → PD 제어 → 토크(정적 clip)". 주석(`config.py:118-121`)에 나오듯, 처음엔 `DCMotor`를 썼다가 그것의 토크-속도 감쇠가 원본에 없던 제약이라 로봇이 **기어다니는 국소최적**에 빠져서 `IdealPD`로 바꿨습니다. (CODE_GUIDE §8의 "기어다니는 학습" 교훈의 원인)

> Go1은 실제 액추에이터 신경망(`ActuatorNetMLPCfg`, `config.py:16`)을 쓰지만 Go2용 신경망이 없어서 명시적 PD로 갑니다.

### 2.5 씬 구성 — `legged_robot.py:561-593`

`_setup_scene()`가 Isaac Lab 씬을 채웁니다:

```python
self.terrain = Terrain(self.cfg.terrain, self.num_envs)  # 지형 생성
self._create_trimesh()                                    # heightfield→trimesh→USD
self._init_robot()                                        # 로봇 배치
self.scene.clone_environments(copy_from_source=False)     # 환경 6144개 복제
self.scene.filter_collisions(...)                         # 충돌 필터링
light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)      # 조명
```

Isaac Lab의 `clone_environments`가 하나의 프로토타입 환경을 GPU상에서 수천 개로 병렬 복제하는 부분입니다.

### 2.6 접촉 센서 — `config.py:143-155`

```python
class ContactSensorSceneCfg(InteractiveSceneCfg):
    contact_forces_FOOT = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", ...)
    contact_forces_THIGH = ContactSensorCfg(...)
    contact_forces_CALF = ContactSensorCfg(...)
```

발/허벅지/정강이별 접촉력 센서. 관측의 발접촉 4차원과 충돌(collision) 보상 계산에 쓰입니다.

**요약**: Isaac Lab 세팅의 축은 ① 앱 런칭 순서(train.py) ② `SimulationCfg`+`PhysxCfg`(시뮬 물리) ③ `ArticulationCfg`(로봇 USD+액추에이터) ④ `_setup_scene`(지형·복제·조명) ⑤ 접촉 센서.

---

## 3. 논문 내용이 어떻게 구현됐는가

Eurekaverse 논문은 두 부분으로 이뤄집니다: **(가) LLM이 지형 커리큘럼을 진화시키는 메타 루프** (논문의 핵심 기여), **(나) 그 위에서 도는 extreme-parkour 학습 방식** (RMA teacher-student + depth 증류). 논문 흐름을 따라가며 "어디를 보면 되는지"를 짚습니다.

### 3.1 문제 설정: 지형을 "코드"로 표현 (논문 §3)

논문의 출발점은 **지형을 자연어가 아니라 실행 가능한 Python 함수로 표현**하는 것입니다. 그 계약이 `eurekaverse/gpt/system_prompt.txt`에 그대로 GPT에게 전달됩니다:

```python
def set_terrain(length, width, field_resolution, difficulty):
    height_field = np.zeros(...)   # 높이맵 (미터)
    goals = np.zeros((8, 2))       # 8개 골 (순서대로 통과)
    return height_field, goals
```

- **보려면**: 프롬프트 계약은 `gpt/system_prompt.txt`, 실제 예시 지형은 `gpt/terrain_example_initial.py`, 최소 구현 예제는 `legged_gym/utils/set_terrain_simple.py`.
- `difficulty` 0~1 인자가 **난이도 커리큘럼의 손잡이**입니다 (예: `gap_width = 0.75 + 0.25*difficulty`).

### 3.2 진화 알고리즘 루프 (논문 §4) — `run_eurekaverse.py:518-597`

논문의 핵심 그림(생성→학습→평가→선택→진화)이 `main()`의 for 루프에 1:1 대응합니다:

```
for it in range(iterations):                     # 논문: 5 iterations
  1. 지형 생성 (병렬 run마다)
     it==0: initial_generation()                 # GPT에게 처음부터 생성 요청
     it>0 : evolution_generation()               # 이전 best 지형 + 성적을 주고 진화 요청
  2. setup_training_all_terrains(it)             # 생성된 지형 파일 배치
  3. ThreadPoolExecutor로 N개 병렬 학습+평가       # 각 run: train.py→evaluate.py 서브프로세스
  4. order_best_runs()                           # ★ 골 도달 수로 best run 선정
  5. best의 lineage를 다음 iteration 진화의 씨앗으로
```

논문 개념 → 코드 매핑:

| 논문 개념 | 코드 위치 |
|---|---|
| 지형을 코드로 생성 | `run_eurekaverse.py:276 initial_generation` → `query_gpt_initial` |
| 진화(evolution): 이전 결과를 피드백으로 | `run_eurekaverse.py:329 evolution_generation` → `query_gpt_evolution` |
| **실행 가능성 필터**(GPT가 짠 함수가 돌아가는지) | `run_eurekaverse.py:241 check_response` — `train.py --check_terrain_feasibility` 서브프로세스로 `"Converting heightmap to trimesh"` 로그 확인 |
| 적합도 평가(fitness) = 골 도달 | `run_eurekaverse.py:107 run_evaluation` → stdout의 `Number of goals reached` 파싱 |
| best 선택 & 계보(lineage) | `run_eurekaverse.py:376 order_best_runs` + `parallel_run_lineage` |
| 진화 스케줄 | 스펙 §14: walk pretrain 1000 iter → 5 iter × (8병렬 × 2000 iter) |

**중요 포인트**: 적합도 신호가 보상이 아니라 **골 도달 수**라는 것. CODE_GUIDE §8의 "보상은 속아도 골 도달은 안 속는다"가 여기 반영돼 있습니다 (`reproduction_spec.md:182-195`의 eval stdout 포맷이 파서와 정확히 일치해야 함).

### 3.3 지형 파이프라인: 코드 → 실제 메시 (논문 §4 구현부) — `terrain_gpt.py`

GPT가 낸 numpy 높이맵이 실제 시뮬 지형이 되기까지:

```
set_terrain() → height_field (numpy)
  → fix_terrain()          # GPT 실수 자동수정 (높이 클램프, 골 위치 보정, 스폰 평탄화)
  → convert_heightfield_to_trimesh()   # 삼각형 메시로 (terrain_gpt.py:312)
  → TrimeshTerrainImporter # USD로 주입 (terrain_gpt.py:72)
  → (300만 삼각형 초과 시 청크 분할 — PhysX 한계 대응)
```

- **그리드/커리큘럼** (`config.py:299-301`): 10행(난이도)×40열(변형) 격자, 셀당 18×4m. `Terrain.make_terrain` / `add_terrain_to_map` (`terrain_gpt.py:202,267`).
- LLM이 짠 함수의 실수를 자동 교정하는 `fix_terrain`이 논문에서 "LLM 생성물의 강건성"을 담당하는 부분입니다.

### 3.4 골 추종 & 보상 (논문의 태스크 정의) — `legged_robot.py`

논문의 parkour 태스크는 "8개 골을 순서대로 통과"입니다:

- **골 추종 로직**: `_update_goals` (`legged_robot.py:307`) — 현재 골 도달 판정, 다음 골 방향(`target_yaw`) 계산. (스펙 §4)
- **보상 14항**: `_get_rewards` (`legged_robot.py:480`) + `_reward_*` 함수들(`legged_robot.py:1251~`). 핵심은 `tracking_goal_vel`(골 방향 전진), `tracking_yaw`(방향 정렬)이고 나머지는 페널티. 스케일은 `RewardScales` (`config.py:444`). `only_positive_rewards=True`. (스펙 §3)
- **종료 조건**: `_get_dones` (`legged_robot.py:382`) — 넘어짐(reset_term) vs 타임아웃/완주(reset_time_out). 이 구분이 PPO 가치 부트스트랩에 영향 (CODE_GUIDE §9 함정 4번).
- **지형 커리큘럼 승급/강등**: `_update_terrain_curriculum` (`legged_robot.py:712`) — 8골 중 ≥80% 도달 시 어려운 행으로 (스펙 §6).

### 3.5 학습 방식: RMA teacher-student (논문이 채택한 extreme-parkour 방식, §4·§5.1)

논문은 지형을 생성하는 게 기여이고, 학습 자체는 extreme-parkour의 방식을 씁니다. 핵심은 **특권정보(privileged) teacher를 배포 가능한 student로 증류**하는 것:

```
관측 698 ─┬─ proprio 48 ──────────────┬──→ Actor MLP → 액션 12
          ├─ scandots 132 → ScanEncoder┤
          ├─ priv_explicit 9 ──────────┤   (추론 시 Estimator 출력으로 대체)
          ├─ priv_latent 29 → PrivEncoder┤  (추론 시 HistoryEncoder 출력으로 대체)
          └─ history 480 → HistoryEncoder┘
```

- **관측 조립**: `_get_observations` (`legged_robot.py:503`). 어느 채널이 특권정보라 배포 시 재구성이 필요한지가 CODE_GUIDE §2 표에 정리돼 있음. (스펙 §1)
- **신경망 구조**: `rsl_rl/modules/actor_critic.py` (`ActorCriticRMA`). 학습 땐 진짜 속도/마찰로 배우고, 배포 땐 그걸 proprio/히스토리에서 **추정**.
- **Estimator**(속도 추정, MSE): `rsl_rl/modules/estimator.py`.
- **HistoryEncoder DAgger**(PrivEncoder 출력 따라하기, `dagger_update_freq=20`): `rsl_rl/algorithms/ppo.py`.
- **PPO 하이퍼파라미터**: `LeggedRobotCfgPPO` (`config.py:537`) — lr 2e-4 adaptive KL, γ0.99, λ0.95, entropy 0.01, 24스텝/iter. (스펙 §14)

### 3.6 depth 증류: scandots → 카메라 (논문 §4 배포 단계) — `distill_eurekaverse.py`

teacher는 특권정보인 scandots(지형 높이 격자)를 쓰지만, 실기 로봇엔 그게 없습니다. 그래서 depth 카메라 이미지에서 scandots를 대체하도록 student를 증류:

- **depth 인코더**: `rsl_rl/modules/depth_backbone.py`.
- **증류 실행**: `eurekaverse/distill_eurekaverse.py`, 설정 `eurekaverse/config_distill/config.yaml`. (스펙 §11)
- 카메라 설정은 `CustomDepthCfg` (`config.py:193`, `use_camera` 플래그).

### 3.7 벤치마크 평가 (논문 §5.2) — `set_terrain_benchmark.py`

논문의 정량 비교는 고정된 20개 테스트 지형 × 10난이도에서 골 도달 수를 측정:

- 벤치마크 지형: `legged_gym/utils/set_terrain_benchmark.py` (순수 numpy, 20태스크).
- 판정 기준: Eurekaverse teacher ≥ original 대조군, iteration 0→4 단조 상승 (스펙 §13).

### 논문 → 코드 한눈 지도

| 논문 파트 | 핵심 파일 | 진입 함수/클래스 |
|---|---|---|
| 지형=코드 표현 (§3) | `gpt/system_prompt.txt`, `set_terrain_*.py` | `set_terrain()` |
| 진화 루프 (§4) | `run_eurekaverse.py` | `main()`, `evolution_generation` |
| 실행가능성 필터 | `run_eurekaverse.py` | `check_response` |
| 지형→메시 | `terrain_gpt.py` | `Terrain`, `fix_terrain` |
| 태스크(골+보상) | `legged_robot.py` | `_update_goals`, `_get_rewards` |
| RMA teacher-student (§4,5.1) | `actor_critic.py`, `ppo.py` | `ActorCriticRMA` |
| depth 증류 | `distill_eurekaverse.py` | — |
| 벤치마크 (§5.2) | `set_terrain_benchmark.py` | — |

가장 좋은 "논문 대조" 참고서는 [reproduction_spec.md](reproduction_spec.md)입니다 — 각 절이 논문 절(§)과 코드 줄번호를 함께 명시하고 있어서, 논문을 펴 놓고 이 문서를 색인 삼아 코드로 내려가면 됩니다.
