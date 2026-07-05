# 직접 진행 가이드 — 현재 상태와 남은 단계

작성 시점 상태: **백엔드 검증 완료** (walk_pretrain이 6.24/8 골 도달, 교정 21건 반영, 커밋 dbe0b17).
지금 돌아가는 프로세스 없음. GPU 비어 있음.

---

## STEP 1. 벤치마크 베이스라인 (10분, 바로 가능)

walk_pretrain을 논문의 20태스크 벤치마크에서 평가 — 재연 테이블의 첫 행이자 파쿠르 학습 전 베이스라인:

```bash
conda activate eureka_lab6
cd ~/workspace/eurekaverse_lab3/extreme-parkour/legged_gym/legged_gym
python -u scripts/evaluate.py --task go2 --exptid walk_pretrain --headless \
    --max_steps 1001 --metric_granularity type --terrain_type benchmark \
    | tee ~/benchmark_walk_pretrain.log
```
- 기대: 걷기만 배운 정책이라 전체 평균 goals는 낮게 (1~3), 쉬운 태스크에서만 점수. **이 수치를 기록**해두세요 — 루프 학습 후와 비교하는 기준입니다.
- 출력의 `STATISTICS FOR TERRAIN TYPE 00~19` = 20개 태스크별 성적.

## STEP 2. LLM 루프 연결 패치 (30분~1시간, 에디터 작업)

루프(`eurekaverse/`)는 포크의 도커 환경용 하드코딩이 남아 있어 **4가지 수정이 필요**합니다:

### 2a. 실행 프리픽스 교체 — `eurekaverse/run_eurekaverse.py`
- `python_prefix = "/isaaclab/isaaclab.sh -p"` 로 검색 → 우리 환경으로:
  ```python
  python_prefix = f"{os.path.expanduser('~')}/miniconda3/envs/eureka_lab6/bin/python"
  ```
- `/isaaclab/workspace/export_vars.sh` 를 읽는 블록(파일 상단) → 통째로 삭제하거나 `if os.path.exists(...)` 가드. 대신 `os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"` 한 줄 추가.

### 2b. 시드 체크포인트 이름 — 같은 파일
- `"flat_action-d0_spd-08"` 검색 → `"walk_pretrain"` 으로 (iteration-0이 우리가 만든 체크포인트에서 시작하도록).

### 2c. config 정렬 — `eurekaverse/config/config.yaml`
```yaml
quadruped_model: "go2"        # 필수
iterations: 5                  # 논문값 (포크는 9)
train_iterations: 2000         # 논문값 (포크는 1500)
num_parallel_runs: 1           # 단일 GPU — 포크가 1-run 흐름을 이미 구현해둠
num_parallel_checks: 4
gpt_model: "gpt-5.4"
wandb: False
# terrain_train / terrain_eval / terrain_benchmark 블록: 고증 백엔드값으로
#   horizontal_scale: 0.05  (포크는 0.1)
#   num_rows: 10            (포크는 8)
#   terrain_train의 num_cols: 40, terrain_benchmark의 num_cols: 20
```
⚠️ 이 블록 값들은 학습 명령에 `--horizontal_scale 0.05 --num_rows 10 ...` 플래그로 붙어 백엔드 config를 덮어씁니다 — 백엔드 기본값과 일치시키는 것이 안전합니다.

### 2d. gpt-5.4 가격 등록 — `eurekaverse/utils/gpt_utils.py`
- `gpt_pricing = {` 딕셔너리에 추가: `"gpt-5.4": (2.5e-6, 15e-6),`
- (포크가 API 호출을 이미 병렬 단건 방식으로 바꿔놔서 `n=` 문제는 없음 — gpt-5.4 호환)
- 안전장치: `gpt_pricing[cfg.gpt_model]` 을 `gpt_pricing.get(cfg.gpt_model, (0.0, 0.0))` 으로.

### 2e. API 키
```bash
echo 'export OPENAI_API_KEY=sk-...' >> ~/.bashrc && source ~/.bashrc
```

## STEP 3. 루프 파일럿 (반나절, ~$5~10 API 비용)

축소판으로 전체 배관 검증:
```bash
# config.yaml에서 임시로: iterations: 2, train_iterations: 500
cd ~/workspace/eurekaverse_lab3/eurekaverse
python run_eurekaverse.py
```
- 출력: `eurekaverse/outputs/.../` 아래 iteration별 지형 파일·평가 로그·피클
- **성공 판정**: 2 iteration 무인 완주 + 로그에 "Best run in iteration" + 지형 파일 생성
- 막히는 지점은 대부분 로그 문자열 대기 타임아웃(20분) — `outputs/.../train_iter-*.log` 를 열어 실제 에러를 보세요

## STEP 4. 프로덕션 루프 (4~6일 연속, ~$50~150 API)

```bash
# config.yaml 복원: iterations: 5, train_iterations: 2000
nohup python run_eurekaverse.py > ~/loop_production.log 2>&1 &
```
- 크래시 시 `resume_run: "<RUN_ID>"` 로 재개
- 완료 후: 최종 정책을 benchmark 평가 → STEP 1의 베이스라인과 비교 → **iteration별 testing goals가 상승 추세**면 논문 Fig 4 재연 성공

## STEP 5. 이후 단계 (필요 시 지원 요청)

| 단계 | 상태 | 비고 |
|---|---|---|
| 증류 (M6) | **이식 필요** | rsl_rl depth 가드 제거, helpers camera 분기 복원, --use_camera/--action_delay CLI, action delay 런타임 — verification_report §9 "M6 항목" 참조. 난이도 높음 |
| jit 내보내기 (M7) | **save_jit.py 미이식** | 원본 `~/workspace/eurekaverse/extreme-parkour/legged_gym/legged_gym/scripts/save_jit.py` 참조 |
| original 대조군 (M9) | 거리 커리큘럼 미구현 | `_update_terrain_curriculum`의 NotImplementedError 부분, 원본 코드 그대로 이식하면 됨 |
| 실기 배포 (M8) | 미착수 | Go2 EDU + D435i 필요, Extreme-Parkour-Onboard 참조 |

## 문제가 생기면

1. **골 도달 지표부터** — 보상 상승은 믿지 말 것 (CODE_GUIDE §8)
2. 진단 스크립트: `scripts/diag_*.py` (스폰/접촉/골신호/뷰포트)
3. 원본 대조: `diff <(sed -n '구간' ~/workspace/eurekaverse/...) <(sed -n '구간' 우리파일)`
4. 되돌리기: `git log --oneline` → `git checkout <커밋> -- <파일>` (baseline = dbe0b17)
5. 각 수치의 근거: reproduction_spec.md / 편차 이력: verification_report.md
