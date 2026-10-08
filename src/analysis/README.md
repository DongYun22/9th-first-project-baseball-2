# 📊 Analysis (분석 로직)

`main.py` 파이프라인이 쓰는 분석 함수들입니다. 계산은 재사용 가능한 순수 함수로 두고, 데이터 로드와 실행 순서는 `src/report/`와 `main.py`가 맡습니다. 순수 함수는 `tests/analysis/`에 단위 테스트가 있습니다. R 연동 코드(`glmer_runner.py`)는 R이 없는 환경도 있어 단위 테스트 대신 실제 데이터로 파이프라인을 돌려 확인합니다.

| 파일 | 내용 | 최종 정리 |
|---|---|---|
| `avoidance_stats.py` | 시즌 구사율, 대조군 층화추출·1:1 매칭, 평균 차이와 신뢰구간 | p.2–5 |
| `situation_matching.py` | 상황 변수와 상황 맞추기 사다리(M0–M3), 균형(SMD) 점검 | p.3 |
| `robust_inference.py` | 대조군 전체를 가중한 순수 감소, 투수 단위 cluster 부트스트랩 | p.3 |
| `mixed_effects_model.py`, `glmer_runner.py` | 혼합효과 모형용 데이터 구성과 R `lme4` 실행·결과 추출 | p.3, p.9–10 |
| `batter_specific.py` | 같은 장타 이벤트 안에서 중간 타자와 재대결 타자에 대한 회피 비교 | p.4 |
| `rematch_execution.py` | 다시 던진 재대결의 코스·구속·무브먼트를 투수 평소 대비 편차로 비교 | p.4–5 |
| `hit_quality.py` | 결과 × 타구 질(xwOBA 4분위, 배럴)별 회피 | p.6 |
| `usage_change_factors.py` | 같은 시즌 내 재대결의 구종 사용 변화를 GBM으로 예측하고 SHAP으로 변수별 영향 확인 | p.7 |
| `selection_value.py`, `selection_score.py` | GBM 선택점수(구종 바꿔 두 번 예측), 경기 전 구사율, 구사율별 차이, 투수 단위 재표집 | p.9–10 |
| `pitcher_tendency.py` | 투수별 회피율과 시즌 간 안정성(ω²) | p.11 |
| `run_same_batter_rematch_analysis.py` | 같은 타자 재대결 이벤트 구성(`build_xbh_rematch`)과 초기 메인 분석 스크립트 | p.2 |
