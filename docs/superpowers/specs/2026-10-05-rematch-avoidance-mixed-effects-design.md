# 설계: 재대결 회피 효과의 이벤트 단위 Mixed-Effects 검증

2026-10-05. 새 노트북(`07_DY_rematch_avoidance_mixed_effects.ipynb`) 설계입니다. 구현 계획은 이 문서 승인 후 따로 씁니다.

## 1. 배경과 질문

노트북 3·4·6은 전부 같은 틀을 썼다: 재대결 이벤트(27,391건)를 투수 439명의 평균(`mean_rvae`)으로 뭉개고, 그 평균을 회피율 3분위로 잘라 147명씩 비교했다. 네 번 다 p값이 0.06~0.25 사이 경계선에 머물렀고, 리서치 에이전트가 그 이유를 진단했다: **이 압축(투수 평균 → 3분위 컷) 과정 자체가 저표본 투수를 양 극단에 몰아넣는 통계적 함정**이었다(회피율 최상·최하위 10명 중 8명이 표본(n_events) 중앙값 이하). 모델을 더 정교하게 만들거나(노트북 4), 분석 단위를 바꾸거나(노트북 6의 투구 단위 재정의), 데이터를 늘려도 이 함정 자체는 고쳐지지 않았다.

> 같은 질문("재대결에서 장타 맞은 구종을 회피하는 것이 유리한가")을, 투수 평균으로 뭉개지 않고 **이벤트 27,391건을 그대로 단위로 삼아, 투수를 랜덤효과로 둔 mixed-effects 모델로 직접 검정한다.**

사용자가 명시한 제약: 이 결과는 "유의하지 않음을 정리해서 보여주는" 보고서가 되어서는 안 된다 — 즉 통계적으로 더 올바른 방법을 썼을 때 실제로 결론이 달라지는지(또는 적어도 지금까지보다 훨씬 더 결정력 있는 답이 나오는지)를 보는 것이 목적이다. 결과가 어느 쪽으로 나올지는 미리 보장할 수 없지만, 설계 자체가 지금까지 지적된 결함을 정면으로 고친다.

## 2. 기존 인프라 재사용

이 프로젝트에는 이미 R `lme4`를 rpy2로 연결하는 mixed-effects 배선이 있다(`src/analysis/glmer_runner.py`, `src/analysis/mixed_effects_model.py`) — 다른 질문(회피 "행동"이 XBH 이후 placebo 대비 늘어나는지, `reused_same_type ~ group * baseline_usage + ... + (0 + group | pitcher)`)에 쓰인 것이지만, R 적합·고정효과 추출·수렴 확인·ICC 계산 로직은 그대로 재사용한다:

- `fit_glmer(combined, formula)` — `lme4::glmer(family=binomial, nAGQ=0, control=glmerControl(optimizer="bobyqa"))`. (E)에 그대로 재사용.
- `extract_fixed_effects_table()`, `check_convergence()` — 그대로 재사용.
- `compute_icc(pitcher_intercept_variance)` — 그대로 재사용(로지스틱 모델의 ICC 공식).
- (D)는 연속형 타깃(RVAE)이라 `glmer`(이항)가 아니라 `lmer`(가우시안)가 필요함 — `glmer_runner.py`에 `fit_lmer(combined, formula)`를 짝 함수로 추가한다(R 코드 문자열은 `lmer(formula, data=model_data)`로 바뀌고, `family`/`glmerControl` 인자가 빠짐). 고정효과 추출은 `extract_fixed_effects_table()`을 그대로 쓸 수 있는지 확인하고(`lmer`의 `summary()` 계수 테이블은 기본적으로 p-value가 없음 — `lmerTest` 패키지가 설치돼 있으면 `library(lmerTest)`로 t-test 기반 p-value를 받고, 없으면 추정치·표준오차·95% CI만 보고한다), 없으면 `extract_fixed_effects_table_lmer()`를 따로 만든다.
- `pitch_family`(기존 `src/preprocessing/pitch_family.py`의 `map_pitch_family`), `EXTRA_BASE_HIT_EVENTS`(`src/preprocessing/build_next_ab_dataset.py`)도 재사용한다.

노트북 3의 GBM(상황+구종+구사율, `pitcher_id`·실행 품질 제외)과 재대결 이벤트셋(`build_xbh_rematch`)은 그대로 재사용해서, (D)의 타깃(RVAE)을 산출한다.

## 3. 데이터 구성

단위: 재대결 이벤트 중 `has_next_ab=True`인 것 전부(노트북 3~6과 달리 **투수당 최소 이벤트 수 하한을 두지 않는다** — mixed-effects의 부분 풀링(partial pooling)이 저표본 투수를 적절히 수축시켜 반영하는 게 이 설계의 핵심이므로, 임의의 포함 기준(≥10건)으로 다시 거르면 지금까지의 함정을 반복하게 된다).

각 행(재대결 이벤트)에 붙일 컬럼:

- `avoided` = `1 - reused_same_type`(`xbh`에서, 1=회피함)
- 통제변수는 **재대결 결정구 자신의 상황**(원래 장타를 맞은 시점의 상황이 아님): `balls`, `strikes`, `outs_when_up`, `score_diff`, `platoon_match`, `pitch_family`(재대결에서 실제로 던진 구종을 `map_pitch_family`로 매핑) — 노트북 3의 `find_rematch_decisive_pitch`로 얻는 테이블에서 가져온다.
- `season` — `factor(season)`로 고정효과에 포함(연도별 전반적 수준 차이를 통제).
- `pitcher` — 랜덤효과 그룹 변수(문자열로 변환, 기존 `build_combined_model_dataset`과 동일한 관례).
- (D) `rvae` = 그 재대결 결정구의 실제 `woba_value` − 노트북 3 GBM의 기대값.
- (E) `allowed_xbh_again` = 재대결 결정구의 `events`가 `EXTRA_BASE_HIT_EVENTS`(double/triple/home_run)에 속하면 1, 아니면 0.

결정구를 못 찾은 재대결(노트북 3에서 0.49%였던 것과 같은 원인: 고의사구로 인한 구종 미분류, 미완결 타석)은 제외하고, 노트북 3과 같은 방식으로 원인별 건수를 보고한다.

## 4. 모델

두 모델 다 고정효과는 동일하고 랜덤효과도 동일하게 시작한다(단순하게 시작 → 수렴 안 되면 랜덤 슬롭 등으로 확장, 기존 프로젝트의 교훈과 동일한 태도):

```
(D) rvae ~ avoided + balls + strikes + outs_when_up + score_diff + platoon_match
           + pitch_family + factor(season) + (1 | pitcher)         [lmer, gaussian]

(E) allowed_xbh_again ~ avoided + balls + strikes + outs_when_up + score_diff + platoon_match
           + pitch_family + factor(season) + (1 | pitcher)         [glmer, binomial]
```

보고 항목(둘 다): `avoided`의 추정치·표준오차·(가능하면) p-value·95% CI, (E)는 추가로 승산비(odds ratio), 수렴 상태(`check_convergence`), ICC(투수 간 분산 비율 — 노트북 3의 ω²=0.266과 같은 성격의 수치라 직접 비교 가능).

랜덤효과 구조가 특이(isSingular)하게 나오면(기존 `run_mixed_effects_model.py`에서 실제로 겪은 문제) `(1|pitcher)`를 유지한 채 보고하고, 로버스트니스 체크로 랜덤효과 없는 일반 회귀(`lm`/`glm`)와 비교해 추정치가 크게 달라지는지만 확인한다 — 전체를 다시 설계하지 않는다.

## 5. 노트북 구조

(1) CSV 읽기, 상황 피처 (노트북 3 재사용) (2) 재대결 이벤트셋 (노트북 3 재사용, 포함 기준 없음) (3) GBM 학습 및 RVAE 산출 (노트북 3 재사용) (4) 모델링 데이터 구성 ((D)/(E) 타깃·통제변수 조립, 결측 처리 보고) (5) `fit_lmer` 추가 (글머 러너 확장) (6) (D) 모델 적합·결과 (7) (E) 모델 적합·결과 (8) 두 모델의 ICC를 노트북 3의 ω²=0.266과 나란히 비교 (9) 해석·한계.

## 6. 완료 기준

- (D), (E) 둘 다 수렴한 모델이 적합되고, `avoided`의 추정치·SE·CI가 출력됨
- 두 모델의 ICC가 출력되고 노트북 3의 ω²과 비교됨
- 결측/제외 건수가 투명하게 보고됨(노트북 3의 "결정구 못 찾음" 원인별 분해와 동일한 수준)
- 노트북이 headless로 처음부터 끝까지 에러 없이 실행됨
- 결과가 유의하든 아니든, 왜 그런지(ICC·통제변수·수렴 상태 포함)까지 설명되어 있음 — "유의하지 않다"만 보고하고 끝나지 않음

## 7. 범위 밖

일반 구종 추천 시스템, 실행 품질 피처, `pitcher_id`를 고정효과로 직접 사용, 랜덤 슬롭 이상의 복잡한 랜덤효과 구조(수렴 문제 생기면 로버스트니스 체크로만 다룸), 노트북 3~6의 재실행·수정.

## 8. 한계

- `avoided`는 재대결 전체에서 "장타 맞은 구종을 한 번도 안 던졌는지"의 이진 요약이라, 재대결이 여러 투구로 이어졌을 때 "몇 번째 투구부터 피했는지" 같은 세부는 반영하지 못한다.
- (D)의 `rvae`는 노트북 3의 GBM(상황+구종+구사율만, R²=0.057)에서 나온 값이라, 그 모델의 한계(실행 품질 미반영, 표본 내 예측 등)를 그대로 물려받는다.
- 투수당 최소 표본 하한을 없앴기 때문에, 재대결이 1~2건뿐인 투수도 포함된다 — mixed-effects의 부분 풀링이 이를 적절히 처리한다고 기대하지만, `pitcher`를 랜덤효과로 둔 모델이 실제로 수렴하는지(isSingular 여부)는 실행해서 확인해야 한다.
- `lmer`은 기본적으로 p-value를 주지 않는다 — `lmerTest`가 설치돼 있는지 먼저 확인하고, 없으면 추정치·CI까지만 보고한다(설치를 요구하지 않는다, 범위 밖).
