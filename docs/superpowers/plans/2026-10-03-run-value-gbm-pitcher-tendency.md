# Run Value GBM 회피 성향 평가 노트북 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** scikit-learn `HistGradientBoostingRegressor`로 상황별 기대 Run Value(woba_value) 모델을 만들고, 장타 재대결 결정구의 실제 결과와 비교(RVAE)해 회피형/승부형 투수 그룹(3분위·2분위)의 장기 결과를 비교하는 노트북을 만든다.

**Architecture:** 노트북 01·02와 같은 패턴 — percent 형식 `.py` 원본을 스크래치패드에 작성해 테스트 먼저 방식으로 검증한 뒤, `to_ipynb.py`로 변환하고 `jupyter nbconvert --execute`로 headless 실행해 `notebooks/03_DY_run_value_gbm_pitcher_tendency.ipynb`를 만든다. 팀 CSV만 읽고(`data/raw`, R 불필요), `src/analysis/run_same_batter_rematch_analysis.py`와 `src/analysis/avoidance_stats.py`의 기존 함수를 재사용한다.

**Tech Stack:** Python 3.14(`.venv`), pandas, numpy, scipy, scikit-learn 1.9(`HistGradientBoostingRegressor`), Jupyter

**Spec:** `docs/superpowers/specs/2026-10-03-run-value-gbm-pitcher-tendency-design.md`

## Global Constraints

- Run value 타깃: `woba_value`(CSV 그대로, 타석을 끝낸 결정구에만 채워짐)
- 레퍼토리(반사실 비교) 기준: 투수·시즌 `season_usage_rate` ≥ 0.05
- 회피율·RVAE 분석 포함 기준: 투수당(2021–2026 합산) 재대결 이벤트 ≥ 10건
- 시즌 안정성 검증 포함 기준: 투수-시즌 재대결 이벤트 ≥ 5건, 자격 시즌 2개 이상인 투수만
- 그룹화: 3분위(하위=승부형/상위=회피형/중위 제외)와 2분위(중앙값) **둘 다** 계산
- GBM 학습/검증 분할: 2021–2025 학습, 2026 검증(MAE, 베이스라인 대비 개선폭). 최종 적용 모델은 전체(2021–2026) 재학습
- `pitcher_id`를 피처로 쓰지 않는다
- `data/raw`, R/rpy2를 쓰지 않는다(CSV만)
- 새 의존성은 `requirements.txt`에 추가(`scikit-learn`)
- 커밋 메시지·PR에 Claude 관련 문구를 넣지 않는다(이 세션의 지속 규칙)

## Review Focus

- 결정구가 없는 재대결 타석(데이터가 시즌 마지막 날짜에서 끊겨 PA가 미완결인 극히 드문 경우) → RVAE 계산에서 조용히 빠지지 않고 몇 건이 빠졌는지 로그로 남아야 함
- 레퍼토리에 구종이 1개뿐인 투수(반사실 비교 대상이 없음) → 에러 대신 그 이벤트를 "비교 불가"로 건너뛰고 건수를 보고해야 함
- 시즌 안정성 검증에서 자격 있는 투수가 너무 적어 분산 분해가 불안정할 가능성 → 자격 투수 수를 함께 출력해 독자가 판단하게 해야 함
- 3분위·2분위 그룹화에서 포함 기준(10건)을 만족하는 투수 수가 두 그룹에 고르지 않게 쏠릴 가능성(예: 회피형이 압도적으로 적음) → 그룹별 투수 수를 항상 함께 출력해야 함
- GBM이 `pitch_type` 범주를 학습 때 못 본 값으로 반사실 대입하면(예: 희귀 구종 코드) 조용히 틀린 예측을 내지 않고 명시적으로 걸러야 함(레퍼토리 5% 기준이 이를 대부분 막지만, 학습 데이터 자체에 아주 희귀한 구종이 전혀 없을 가능성도 있음)

---

### Task 0: 환경 — scikit-learn 추가

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: requirements.txt에 scikit-learn 추가**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && python3 - <<'EOF'
from pathlib import Path
path = Path("requirements.txt")
lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
if "scikit-learn" not in lines:
    lines.append("scikit-learn")
path.write_text("\n".join(lines) + "\n")
print(path.read_text())
EOF
```
Expected: 기존 10줄 + `scikit-learn` 11줄이 출력된다.

- [ ] **Step 2: 설치 확인 및 pytest**

Run: `.venv/bin/python -c "import sklearn; print(sklearn.__version__)" && .venv/bin/python -m pytest -q 2>&1 | tail -2`
Expected: `1.9.1`(또는 그 이상), `86 passed`

---

### Task 1: 노트북 스캐폴드 + CSV 읽기

**Files:**
- Create: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py` (`$NB_SRC` = `/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src`)
- Verify/recreate: `$NB_SRC/to_ipynb.py` (세션 스크래치패드가 정리됐을 수 있음)

**Interfaces:**
- Produces: `pitches`(DataFrame, 전체 표본 + 추가 컬럼), `LOAD_COLUMNS`

- [ ] **Step 1: to_ipynb.py 존재 확인, 없으면 재생성**

Run: `test -f $NB_SRC/to_ipynb.py && echo OK || echo MISSING`
Expected: `OK`이면 다음 단계로. `MISSING`이면 아래로 재생성한다(노트북 01/02 때 쓴 것과 동일한 내용).

```bash
mkdir -p $NB_SRC && cat > $NB_SRC/to_ipynb.py <<'PYEOF'
"""percent 형식 .py -> .ipynb (nbformat 4). 사용: python to_ipynb.py SRC.py OUT.ipynb"""
import re, sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

MARKER = re.compile(r"^# %%(?P<md> \[markdown\])?(?P<title>.*)$")
MAGIC = re.compile(r"^# (%[A-Za-z].*)$")


def convert(source_text: str):
    cells, kind, buffer = [], None, []
    def flush():
        body = "\n".join(buffer).strip("\n")
        if kind is None or not body.strip():
            return
        if kind == "markdown":
            cells.append(new_markdown_cell("\n".join(re.sub(r"^# ?", "", l) for l in body.split("\n"))))
        else:
            cells.append(new_code_cell("\n".join(MAGIC.sub(r"\1", l) for l in body.split("\n"))))
    for line in source_text.split("\n"):
        m = MARKER.match(line)
        if m:
            flush(); kind = "markdown" if m.group("md") else "code"; buffer = []
        else:
            buffer.append(line)
    flush()
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    nbformat.write(convert(src.read_text(encoding="utf-8")), out)
    print(f"wrote {out}")
PYEOF
```

- [ ] **Step 2: 제목·설정 셀 작성**

Run:
```bash
cat > $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'
# %% [markdown]
# # ⚾ Run Value 기반 GBM: 회피형/승부형 투수의 판단은 장기적으로 좋은 선택이었나
#
# * **작성자:** DY
# * **작성일:** 2026-10-03
# * **목적:**
#   * 장타를 맞은 구종을 재대결에서 회피하는 것이 투수에게 Run Value 기준 실제 유리한 선택인지 봅니다.
#   * GBM은 "이 한 타석에 뭘 던졌어야 했나"를 추천하지 않고, 상황별 기대 Run Value를 계산하는 보정 기준선으로만 씁니다.
#     그 기준선 대비 실제 결과(RVAE)를 투수별로 모아 회피율 그룹(3분위/2분위) 간 평균을 비교하는 것이 메인 결과입니다.
#   * 개별 타석에서 레퍼토리 중 최선이었는지 보는 반사실 비교는 보조 지표입니다.
# * **설계 문서:** `docs/superpowers/specs/2026-10-03-run-value-gbm-pitcher-tendency-design.md`

# %%
# 1. 경로 설정 (src 폴더 접근용)
import sys, os
sys.path.append(os.path.abspath('..'))

# 2. Autoreload 설정
# %load_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.analysis.avoidance_stats import compute_season_usage_rate, summarize_diff_in_diff
from src.analysis.run_same_batter_rematch_analysis import build_xbh_rematch
from src.preprocessing.research_sample import KEY_COLUMNS, list_research_csvs

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 60)

MIN_USAGE_RATE = 0.05
MIN_REMATCH_EVENTS = 10
MIN_SEASON_EVENTS = 5
TRAIN_SEASONS = (2021, 2022, 2023, 2024, 2025)
VALID_SEASON = 2026
RANDOM_STATE = 20261003

# %% [markdown]
# ## 1. 팀 CSV 읽기
#
# `data/raw` 없이 팀 research CSV 6개를 읽습니다. `woba_value`/`woba_denom`은 그 타석을 끝낸 결정구에만 채워져 있고
# (실측 확인됨), 이게 GBM의 학습·평가 단위(타석이 아니라 결정구 하나)를 정합니다.

# %%
LOAD_COLUMNS = [
    'game_pk', 'game_date', 'at_bat_number', 'pitch_number', 'pitcher', 'player_name', 'batter',
    'stand', 'p_throws', 'pitch_type', 'events', 'balls', 'strikes', 'outs_when_up',
    'inning', 'inning_topbot', 'on_1b', 'on_2b', 'on_3b', 'home_score', 'away_score',
    'woba_value', 'woba_denom',
]


def load_team_csv() -> pd.DataFrame:
    frames = [
        pd.read_csv(path, usecols=LOAD_COLUMNS, encoding='utf-8-sig', low_memory=False)
        for path in list_research_csvs()
    ]
    pitches = pd.concat(frames, ignore_index=True)
    pitches['season'] = pd.to_datetime(pitches['game_date']).dt.year
    return pitches


pitches = load_team_csv()
assert len(pitches) == 3_592_302 and pitches['pitcher'].nunique() == 1_067
assert not pitches.duplicated(KEY_COLUMNS).any()
print(f'{len(pitches):,}행, 투수 {pitches["pitcher"].nunique():,}명')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py
```
Expected: `3,592,302행, 투수 1,067명`. 에러 없음.

---

### Task 2: 상황 피처 (주자·점수차·동타 여부)

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

**Interfaces:**
- Produces: `add_situation_columns(df) -> df`(컬럼: `runners_n`, `risp`, `score_diff`, `platoon_match` 추가), `SITUATION_COLUMNS`

- [ ] **Step 1: 테스트 먼저 작성**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% test: situation columns
_t = pd.DataFrame({
    'on_1b': [np.nan, 1.0, np.nan], 'on_2b': [np.nan, np.nan, 2.0], 'on_3b': [np.nan, np.nan, np.nan],
    'inning_topbot': ['Top', 'Bot', 'Top'], 'home_score': [3, 5, 2], 'away_score': [3, 2, 4],
    'stand': ['L', 'R', 'R'], 'p_throws': ['R', 'R', 'L'],
})
_r = add_situation_columns(_t)
assert _r['runners_n'].tolist() == [0, 1, 1]
assert _r['risp'].tolist() == [0, 0, 1]
assert _r['score_diff'].tolist() == [0, 3, -2]
assert _r['platoon_match'].tolist() == [0.0, 1.0, 0.0]
print('add_situation_columns 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -3
```
Expected: FAIL with `NameError: name 'add_situation_columns' is not defined`

- [ ] **Step 2: 구현 셀을 테스트 셀 바로 앞에 삽입**

Edit tool로 `old_string`을 `# %% test: situation columns`로, `new_string`을 아래로 바꾼다:

```python
# %% [markdown]
# ## 2. 상황 피처
#
# `score_diff`는 `src/preprocessing/build_next_ab_dataset.py`의 `compute_score_diff()`와 같은 정의(투수팀 관점,
# 수비팀 − 공격팀)를 벡터화한 것입니다.

# %% situation columns
SITUATION_COLUMNS = ['runners_n', 'risp', 'score_diff', 'platoon_match']


def add_situation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    on1, on2, on3 = out['on_1b'].notna(), out['on_2b'].notna(), out['on_3b'].notna()
    out['runners_n'] = on1.astype(int) + on2.astype(int) + on3.astype(int)
    out['risp'] = (on2 | on3).astype(int)
    out['score_diff'] = np.where(
        out['inning_topbot'] == 'Top', out['home_score'] - out['away_score'], out['away_score'] - out['home_score']
    )
    out['platoon_match'] = np.where(
        out['stand'].isna() | out['p_throws'].isna(), np.nan, (out['stand'] == out['p_throws']).astype(float)
    )
    return out


# %% test: situation columns
```

- [ ] **Step 3: 실제 데이터에 적용**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %%
pitches = add_situation_columns(pitches)
print(pitches[SITUATION_COLUMNS].describe().round(3))
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -14
```
Expected: `add_situation_columns 테스트 통과` 이후 describe 표. 에러 없음.

---

### Task 3: 재대결 이벤트 + 투수별 회피율(시즌별·합산)

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

**Interfaces:**
- Consumes: `build_xbh_rematch(pitches, season_usage)`(기존 src 함수, 반환 컬럼에 `pitcher`, `season`, `reused_same_type`, `has_next_ab` 포함)
- Produces: `xbh`(이벤트 DataFrame), `season_rate(events) -> DataFrame[pitcher, season, n_events, avoidance_rate]`, `pitcher_rate(events) -> DataFrame[pitcher, n_events, avoidance_rate]`

- [ ] **Step 1: 재대결 이벤트 생성 + 건수 검증**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% [markdown]
# ## 3. 재대결 이벤트와 투수별 회피율
#
# 장타 재대결 이벤트는 노트북 01·02와 같은 건수(27,391건)가 나와야 합니다. 회피율 = 1 − 평균(`reused_same_type`).

# %%
season_usage = compute_season_usage_rate(pitches)
xbh = build_xbh_rematch(pitches, season_usage)
xbh = xbh[xbh['has_next_ab']].copy()
assert len(xbh) == 27_391, len(xbh)
print(f'재대결 이벤트 {len(xbh):,}건')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && (time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py) 2>&1 | tail -6
```
Expected: `재대결 이벤트 27,391건`. 1~2분 소요.

- [ ] **Step 2: 회피율 집계 함수 테스트 먼저**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% test: avoidance rate aggregation
_e = pd.DataFrame({
    'pitcher': [1, 1, 1, 2, 2], 'season': [2021, 2021, 2022, 2021, 2021],
    'reused_same_type': [1, 0, 0, 1, 1],
})
_s = season_rate(_e)
assert _s.set_index(['pitcher', 'season'])['n_events'].to_dict() == {(1, 2021): 2, (1, 2022): 1, (2, 2021): 2}
assert np.isclose(_s.set_index(['pitcher', 'season']).loc[(1, 2021), 'avoidance_rate'], 0.5)
_p = pitcher_rate(_e)
assert _p.set_index('pitcher')['n_events'].to_dict() == {1: 3, 2: 2}
assert np.isclose(_p.set_index('pitcher').loc[1, 'avoidance_rate'], 1 - 1 / 3)
assert np.isclose(_p.set_index('pitcher').loc[2, 'avoidance_rate'], 0.0)
print('회피율 집계 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -3
```
Expected: FAIL with `NameError: name 'season_rate' is not defined`

- [ ] **Step 3: 구현 삽입**

`old_string`: `# %% test: avoidance rate aggregation`
`new_string`:
```python
# %% avoidance rate aggregation
def _agg_rate(events: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = events.groupby(group_cols)['reused_same_type']
    out = g.agg(n_events='size', reuse_rate='mean').reset_index()
    out['avoidance_rate'] = 1 - out['reuse_rate']
    return out.drop(columns='reuse_rate')


def season_rate(events: pd.DataFrame) -> pd.DataFrame:
    return _agg_rate(events, ['pitcher', 'season'])


def pitcher_rate(events: pd.DataFrame) -> pd.DataFrame:
    return _agg_rate(events, ['pitcher'])


# %% test: avoidance rate aggregation
```

- [ ] **Step 4: 실행 확인**

Run: `cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -3`
Expected: `회피율 집계 테스트 통과`

---

### Task 4: 시즌 안정성 검증 (η² 분산 분해)

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

**Interfaces:**
- Consumes: `season_rate`, `MIN_SEASON_EVENTS`
- Produces: `eta_squared(df, group_col, value_col) -> float`

- [ ] **Step 1: 테스트 먼저 (완전히 안정적인 경우 η²=1, 완전히 무작위인 경우 η²≈0)**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% test: eta squared
_stable = pd.DataFrame({'pitcher': [1, 1, 2, 2], 'rate': [0.8, 0.8, 0.2, 0.2]})
assert np.isclose(eta_squared(_stable, 'pitcher', 'rate'), 1.0)

_noisy = pd.DataFrame({'pitcher': [1, 1, 2, 2], 'rate': [0.8, 0.2, 0.8, 0.2]})
assert np.isclose(eta_squared(_noisy, 'pitcher', 'rate'), 0.0)
print('eta_squared 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -3
```
Expected: FAIL with `NameError: name 'eta_squared' is not defined`

- [ ] **Step 2: 구현 삽입**

`old_string`: `# %% test: eta squared`
`new_string`:
```python
# %% [markdown]
# ## 4. 시즌 안정성 검증
#
# 회피율을 2021–2026 전체 합산으로 다루기 전에, 투수-시즌 회피율(자격: 그 시즌 재대결 5건 이상, 2개 시즌 이상 자격)이
# 얼마나 그 투수의 안정적 특성인지 확인합니다. η² = (투수 간 분산) / (전체 분산). 진행 여부와 무관하게 보고용입니다.

# %% eta squared
def eta_squared(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    grand_mean = df[value_col].mean()
    total_ss = ((df[value_col] - grand_mean) ** 2).sum()
    if total_ss == 0:
        return float('nan')
    group_means = df.groupby(group_col)[value_col].transform('mean')
    between_ss = ((group_means - grand_mean) ** 2).sum()
    # transform('mean')은 행마다 그 그룹 평균을 반복하므로, 그룹별 가중합이 되도록 그대로 합산해도
    # 각 행이 자기 그룹 평균과의 편차 제곱합에 기여하는 구조(중복 합산 아님)
    return float(between_ss / total_ss)


# %% test: eta squared
```

- [ ] **Step 3: 실제 데이터 적용**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %%
season_rates = season_rate(xbh)
season_rates = season_rates[season_rates['n_events'] >= MIN_SEASON_EVENTS]
multi_season = season_rates[season_rates.groupby('pitcher')['pitcher'].transform('size') >= 2]
print(f'시즌당 {MIN_SEASON_EVENTS}건 이상, 2개 시즌 이상 자격인 투수 {multi_season["pitcher"].nunique()}명 '
      f'({len(multi_season)}개 투수-시즌)')
eta2 = eta_squared(multi_season, 'pitcher', 'avoidance_rate')
print(f'eta-squared (투수 간 분산 비율): {eta2:.3f}')

pairs = (
    multi_season.sort_values(['pitcher', 'season'])
    .assign(next_rate=lambda d: d.groupby('pitcher')['avoidance_rate'].shift(-1),
            next_season=lambda d: d.groupby('pitcher')['season'].shift(-1))
    .dropna(subset=['next_rate'])
)
pairs = pairs[pairs['next_season'] == pairs['season'] + 1]
corr = pairs['avoidance_rate'].corr(pairs['next_rate'])
print(f'인접 시즌 회피율 상관계수 (n={len(pairs)} 쌍): {corr:.3f}')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -6
```
Expected: `eta_squared 테스트 통과`, 자격 투수 수, `eta-squared: 0.xxx`, 인접 시즌 상관계수. 에러 없음. 숫자가 무엇이든(낮아도) 다음 태스크로 진행한다(설계 문서 3.2절).

---

### Task 5: 포함 기준 + 3분위/2분위 그룹화

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

**Interfaces:**
- Consumes: `pitcher_rate`, `MIN_REMATCH_EVENTS`
- Produces: `assign_tercile(df) -> df`(컬럼 `tercile_group` 추가, 값 `'회피형'/'중립'/'승부형'`), `assign_median(df) -> df`(컬럼 `median_group` 추가, 값 `'회피형'/'승부형'`)

- [ ] **Step 1: 테스트 먼저**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% test: grouping
_r = pd.DataFrame({'pitcher': range(6), 'avoidance_rate': [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]})
_t = assign_tercile(_r)
assert _t.set_index('pitcher')['tercile_group'].to_dict() == {
    0: '승부형', 1: '승부형', 2: '중립', 3: '중립', 4: '회피형', 5: '회피형',
}
_m = assign_median(_r)
assert set(_m.loc[_m['avoidance_rate'] <= 0.3, 'median_group']) == {'승부형'}
assert set(_m.loc[_m['avoidance_rate'] >= 0.7, 'median_group']) == {'회피형'}
print('그룹화 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -3
```
Expected: FAIL with `NameError: name 'assign_tercile' is not defined`

- [ ] **Step 2: 구현 삽입**

`old_string`: `# %% test: grouping`
`new_string`:
```python
# %% [markdown]
# ## 5. 포함 기준과 그룹화
#
# 2021–2026 합산 재대결이 10건 이상인 투수만 포함합니다. 3분위(하위=승부형/상위=회피형)와
# 2분위(중앙값)를 모두 만들어 이후 분석을 두 번 반복합니다.

# %% grouping
def assign_tercile(rates: pd.DataFrame) -> pd.DataFrame:
    out = rates.copy()
    q1, q2 = out['avoidance_rate'].quantile([1 / 3, 2 / 3])
    out['tercile_group'] = np.select(
        [out['avoidance_rate'] <= q1, out['avoidance_rate'] >= q2], ['승부형', '회피형'], default='중립'
    )
    return out


def assign_median(rates: pd.DataFrame) -> pd.DataFrame:
    out = rates.copy()
    med = out['avoidance_rate'].median()
    out['median_group'] = np.where(out['avoidance_rate'] >= med, '회피형', '승부형')
    return out


# %% test: grouping
```

- [ ] **Step 3: 실제 데이터 적용**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %%
rates = pitcher_rate(xbh)
rates = rates[rates['n_events'] >= MIN_REMATCH_EVENTS].copy()
rates = assign_median(assign_tercile(rates))
print(f'포함 기준(재대결 {MIN_REMATCH_EVENTS}건 이상) 만족 투수: {len(rates)}명')
print('3분위 그룹별 투수 수:', rates['tercile_group'].value_counts().to_dict())
print('2분위 그룹별 투수 수:', rates['median_group'].value_counts().to_dict())
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -6
```
Expected: `그룹화 테스트 통과`, 포함 투수 수, 그룹별 투수 수(3분위 세 그룹, 2분위 두 그룹). 에러 없음.

---

### Task 6: GBM 학습 테이블 (전체 표본의 결정구)

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

**Interfaces:**
- Produces: `build_decisive_pitch_table(pitches, season_usage) -> DataFrame`(컬럼: `pitch_type`(category), `stand`(category), `p_throws`(category), `outs_when_up`, `runners_n`, `risp`, `score_diff`, `platoon_match`, `season_usage_rate`, `season`, `woba_value`), `MODEL_FEATURES`, `CATEGORICAL_FEATURES`

- [ ] **Step 1: 테스트 먼저**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% test: decisive pitch table
_px = pd.DataFrame({
    'pitcher': [1, 1, 1], 'season': [2021, 2021, 2021], 'pitch_type': ['SL', 'FF', 'SL'],
    'events': [None, None, 'single'], 'woba_value': [np.nan, np.nan, 0.9], 'woba_denom': [np.nan, np.nan, 1.0],
    'outs_when_up': [0, 0, 0], 'runners_n': [0, 0, 0], 'risp': [0, 0, 0], 'score_diff': [0, 0, 0],
    'platoon_match': [1.0, 1.0, 1.0], 'stand': ['R', 'R', 'R'], 'p_throws': ['R', 'R', 'R'],
})
_su = pd.DataFrame({'pitcher': [1, 1], 'season': [2021, 2021], 'pitch_type': ['SL', 'FF'], 'season_usage_rate': [0.6, 0.4]})
_tbl = build_decisive_pitch_table(_px, _su)
assert len(_tbl) == 1  # 결정구(events 비어있지 않음)만 1건
assert _tbl.iloc[0]['pitch_type'] == 'SL' and np.isclose(_tbl.iloc[0]['season_usage_rate'], 0.6)
assert str(_tbl['pitch_type'].dtype) == 'category' and str(_tbl['stand'].dtype) == 'category'
print('build_decisive_pitch_table 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -3
```
Expected: FAIL with `NameError: name 'build_decisive_pitch_table' is not defined`

- [ ] **Step 2: 구현 삽입**

`old_string`: `# %% test: decisive pitch table`
`new_string`:
```python
# %% [markdown]
# ## 6. GBM 학습 테이블
#
# 표본 전체(재대결에 국한하지 않음)의 모든 결정구에서, 그 결정구의 구종이 그 투수·시즌에서 차지하는
# `season_usage_rate`를 붙입니다. `pitcher_id`는 피처에 넣지 않습니다.

# %% decisive pitch table
MODEL_FEATURES = [
    'pitch_type', 'stand', 'p_throws', 'outs_when_up', 'runners_n', 'risp', 'score_diff', 'season_usage_rate',
]
CATEGORICAL_FEATURES = ['pitch_type', 'stand', 'p_throws']


def build_decisive_pitch_table(pitches_sit: pd.DataFrame, season_usage_: pd.DataFrame) -> pd.DataFrame:
    decisive = pitches_sit[pitches_sit['woba_value'].notna()].copy()
    decisive = decisive.merge(
        season_usage_[['pitcher', 'season', 'pitch_type', 'season_usage_rate']],
        on=['pitcher', 'season', 'pitch_type'], how='left',
    )
    for col in CATEGORICAL_FEATURES:
        decisive[col] = decisive[col].astype('category')
    return decisive


# %% test: decisive pitch table
```

- [ ] **Step 3: 실제 데이터 적용**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %%
decisive_table = build_decisive_pitch_table(pitches, season_usage)
assert decisive_table['season_usage_rate'].notna().mean() > 0.99, '구사율 매칭 실패 비율이 높음'
print(f'결정구 {len(decisive_table):,}건, season_usage_rate 결측 {decisive_table["season_usage_rate"].isna().sum()}건')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && (time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py) 2>&1 | tail -5
```
Expected: `build_decisive_pitch_table 테스트 통과`, 결정구 약 90만~100만 건, 구사율 결측 1% 미만. 결측이 1%를 넘으면 멈추고 원인(구종이 없는 결정구, 즉 `pitch_type`이 빈 투구가 PA를 끝낸 극소수 사례 등)을 보고한다.

---

### Task 7: GBM 학습·검증

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

**Interfaces:**
- Produces: `fit_run_value_model(table, seasons) -> HistGradientBoostingRegressor`, `final_model`(전체 데이터로 재학습된 모델)

- [ ] **Step 1: 학습/검증 분할 및 성능 확인**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% [markdown]
# ## 7. GBM 학습과 검증
#
# 2021–2025로 학습, 2026으로 검증합니다. 베이스라인(학습 세트 평균으로만 예측)과 MAE를 비교합니다.

# %% fit run value model
def fit_run_value_model(table: pd.DataFrame, seasons) -> HistGradientBoostingRegressor:
    train = table[table['season'].isin(seasons)]
    model = HistGradientBoostingRegressor(
        categorical_features='from_dtype', random_state=RANDOM_STATE, max_iter=200, early_stopping=True,
    )
    model.fit(train[MODEL_FEATURES], train['woba_value'])
    return model


# %%
train_set = decisive_table[decisive_table['season'].isin(TRAIN_SEASONS)]
valid_set = decisive_table[decisive_table['season'] == VALID_SEASON]
model_cv = fit_run_value_model(decisive_table, TRAIN_SEASONS)
pred_valid = model_cv.predict(valid_set[MODEL_FEATURES])
baseline_mae = mean_absolute_error(valid_set['woba_value'], np.full(len(valid_set), train_set['woba_value'].mean()))
model_mae = mean_absolute_error(valid_set['woba_value'], pred_valid)
print(f'베이스라인(평균 예측) MAE: {baseline_mae:.4f}')
print(f'GBM MAE: {model_mae:.4f} (개선폭 {(baseline_mae - model_mae) / baseline_mae:.1%})')
assert model_mae < baseline_mae, 'GBM이 베이스라인보다 못함 -- 피처/학습을 재검토해야 함'

final_model = fit_run_value_model(decisive_table, decisive_table['season'].unique())
print('전체 데이터로 재학습한 최종 모델 준비 완료')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && (time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py) 2>&1 | tail -8
```
Expected: 베이스라인 MAE와 GBM MAE, 개선폭(양수)이 출력된다. `assert`가 실패하면 멈추고 사용자에게 보고한다(피처 재검토 필요). 수 분 소요 가능(전체 재학습 포함).

---

### Task 8: 재대결 결정구 RVAE

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

**Interfaces:**
- Consumes: `final_model.predict(df[MODEL_FEATURES]) -> np.ndarray`
- Produces: `find_rematch_decisive_pitch(xbh, pitches_sit) -> DataFrame`(컬럼: 이벤트 키 + 결정구의 `MODEL_FEATURES` + `woba_value`), `compute_rvae(df, model) -> Series`

- [ ] **Step 1: 테스트 먼저 (가짜 모델로 RVAE 산술 검증)**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% test: rvae arithmetic
class _StubModel:
    def predict(self, X):
        return np.full(len(X), 0.3)

_df = pd.DataFrame({'woba_value': [0.9, 0.0], **{c: [0] * 2 for c in MODEL_FEATURES}})
_rvae = compute_rvae(_df, _StubModel())
assert np.allclose(_rvae.to_numpy(), [0.6, -0.3])
print('compute_rvae 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -3
```
Expected: FAIL with `NameError: name 'compute_rvae' is not defined`(`find_rematch_decisive_pitch`도 아직 없지만 이 테스트는 그 함수를 안 씀)

- [ ] **Step 2: 구현 삽입**

`old_string`: `# %% test: rvae arithmetic`
`new_string`:
```python
# %% [markdown]
# ## 8. 재대결 결정구 RVAE
#
# 재대결 타석(이벤트의 `next_at_bat_number`)의 결정구를 찾아 모델 기대값과 비교합니다.
# `RVAE = 실제 woba_value − 모델 기대 Run Value` (낮을수록 투수에게 유리).

# %% rematch decisive pitch + rvae
def find_rematch_decisive_pitch(xbh_: pd.DataFrame, pitches_sit: pd.DataFrame) -> pd.DataFrame:
    decisive = pitches_sit[pitches_sit['woba_value'].notna()][
        ['game_pk', 'pitcher', 'at_bat_number', *MODEL_FEATURES, 'woba_value']
    ]
    keys = xbh_[['game_pk', 'pitcher', 'next_at_bat_number']].rename(columns={'next_at_bat_number': 'at_bat_number'})
    out = keys.merge(decisive, on=['game_pk', 'pitcher', 'at_bat_number'], how='left', validate='many_to_one')
    return out


def compute_rvae(df: pd.DataFrame, model) -> pd.Series:
    expected = model.predict(df[MODEL_FEATURES])
    return df['woba_value'] - expected


# %% test: rvae arithmetic
```

- [ ] **Step 3: 실제 데이터 적용 + 투수별 집계**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %%
rematch_decisive = find_rematch_decisive_pitch(xbh, pitches)
missing = rematch_decisive['woba_value'].isna().sum()
print(f'재대결 {len(rematch_decisive):,}건 중 결정구를 못 찾은 건수(타석 미완결 등): {missing} ({missing / len(rematch_decisive):.2%})')
rematch_decisive = rematch_decisive.dropna(subset=['woba_value'])
rematch_decisive['rvae'] = compute_rvae(rematch_decisive, final_model)

pitcher_rvae = rematch_decisive.groupby('pitcher')['rvae'].mean().rename('mean_rvae').reset_index()
rates = rates.merge(pitcher_rvae, on='pitcher', how='left')
assert rates['mean_rvae'].notna().all(), '포함 기준을 만족한 투수 중 RVAE가 없는 경우가 있음'
display(rates.describe().round(4))
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && (time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py) 2>&1 | tail -10
```
Expected: `compute_rvae 테스트 통과`, 결정구 못 찾은 비율(낮아야 함, 대부분 0%대), `rates` describe 표에 `mean_rvae` 컬럼 포함, 에러 없음.

---

### Task 9: 그룹 비교 (메인 결과)

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

- [ ] **Step 1: 3분위·2분위 그룹 간 mean_rvae 비교**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% [markdown]
# ## 9. 그룹 비교 (메인 결과)
#
# 회피형 vs 승부형의 평균 RVAE를 비교합니다(Welch t-test / Mann-Whitney, 기존 `summarize_diff_in_diff` 재사용).
# RVAE가 낮을수록(기대보다 Run Value를 덜 내줬을수록) 유리합니다.

# %%
def compare_groups(df: pd.DataFrame, group_col: str, a: str, b: str) -> dict:
    ga, gb = df.loc[df[group_col] == a, 'mean_rvae'], df.loc[df[group_col] == b, 'mean_rvae']
    r = summarize_diff_in_diff(ga, gb, alternative='two-sided')
    return {'a': a, 'b': b, 'n_a': r['n_treatment'], 'n_b': r['n_control'],
            'mean_a': r['treatment_mean'], 'mean_b': r['control_mean'],
            'diff(a-b)': r['net_effect'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'], 't_p': r['t_p']}


print('[3분위] 회피형 vs 승부형')
display(pd.DataFrame([compare_groups(rates, 'tercile_group', '회피형', '승부형')]).round(4))
print('[2분위] 회피형 vs 승부형')
display(pd.DataFrame([compare_groups(rates, 'median_group', '회피형', '승부형')]).round(4))
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -10
```
Expected: 두 표(3분위, 2분위)가 평균·차이·신뢰구간·p-value와 함께 출력된다. `summarize_diff_in_diff`가 `alternative='two-sided'`를 지원하지 않으면(기존 구현이 `'greater'`/`'less'`만 받을 수 있음) 그 시그니처를 확인해 맞는 인자로 고친다 — `src/analysis/avoidance_stats.py`의 `summarize_diff_in_diff` 정의를 먼저 읽고 실제 지원하는 `alternative` 값으로 맞춘다.

---

### Task 10: 레퍼토리 반사실 비교 (보조 지표)

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`

**Interfaces:**
- Produces: `build_pitcher_repertoire(season_usage_, min_rate) -> dict[(pitcher, season), list[str]]`, `repertoire_gap(row, repertoire, model) -> float | None`

- [ ] **Step 1: 테스트 먼저**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% test: repertoire gap
_su2 = pd.DataFrame({
    'pitcher': [1, 1, 1], 'season': [2021, 2021, 2021], 'pitch_type': ['SL', 'FF', 'CU'],
    'season_usage_rate': [0.5, 0.45, 0.02],
})
_repertoire = build_pitcher_repertoire(_su2, MIN_USAGE_RATE)
assert _repertoire[(1, 2021)] == ['SL', 'FF']  # CU는 2% < 5% 기준으로 제외

class _StubModel2:
    def predict(self, X):
        return np.where(X['pitch_type'].to_numpy() == 'SL', 0.5, 0.2)

_row = pd.Series({'pitcher': 1, 'season': 2021, 'pitch_type': 'SL', 'stand': 'R', 'p_throws': 'R',
                   'outs_when_up': 0, 'runners_n': 0, 'risp': 0, 'score_diff': 0, 'season_usage_rate': 0.5})
_gap = repertoire_gap(_row, _repertoire, _StubModel2())
assert np.isclose(_gap, 0.3)  # 실제(SL=0.5) - 레퍼토리 최선(FF=0.2)
print('repertoire_gap 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -3
```
Expected: FAIL with `NameError: name 'build_pitcher_repertoire' is not defined`

- [ ] **Step 2: 구현 삽입**

`old_string`: `# %% test: repertoire gap`
`new_string`:
```python
# %% [markdown]
# ## 10. 보조 지표: 레퍼토리 반사실 비교
#
# 같은 상황에서 그 투수·시즌의 레퍼토리(구사율 5% 이상 구종) 안의 각 구종을 대입해 예측하고,
# 실제 선택이 레퍼토리 최선과 얼마나 차이 나는지(gap) 봅니다. 레퍼토리가 1개 구종뿐이면 비교 불가(`None`).

# %% repertoire counterfactual
def build_pitcher_repertoire(season_usage_: pd.DataFrame, min_rate: float) -> dict:
    qualified = season_usage_[season_usage_['season_usage_rate'] >= min_rate]
    return qualified.groupby(['pitcher', 'season'])['pitch_type'].apply(list).to_dict()


def repertoire_gap(row: pd.Series, repertoire: dict, model) -> float | None:
    types = repertoire.get((row['pitcher'], row['season']), [])
    if len(types) < 2:
        return None
    candidates = pd.DataFrame([row[MODEL_FEATURES].to_dict()] * len(types))
    candidates['pitch_type'] = types
    for col in CATEGORICAL_FEATURES:
        candidates[col] = candidates[col].astype('category')
    preds = model.predict(candidates[MODEL_FEATURES])
    actual_pred = model.predict(pd.DataFrame([row[MODEL_FEATURES].to_dict()]))[0]
    return float(actual_pred - preds.min())


# %% test: repertoire gap
```

- [ ] **Step 3: 실제 데이터 적용 + 투수별 집계 + 그룹 표에 합치기**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %%
pitcher_repertoire = build_pitcher_repertoire(season_usage, MIN_USAGE_RATE)
gaps = rematch_decisive.apply(lambda r: repertoire_gap(r, pitcher_repertoire, final_model), axis=1)
rematch_decisive['repertoire_gap'] = gaps
not_comparable = gaps.isna().sum()
print(f'레퍼토리 비교 불가(구종 1개 이하) {not_comparable}건 / 전체 {len(gaps)}건')
rematch_decisive['matched_best'] = np.isclose(rematch_decisive['repertoire_gap'].fillna(-1), 0.0)

pitcher_gap = rematch_decisive.groupby('pitcher').agg(
    mean_repertoire_gap=('repertoire_gap', 'mean'), best_match_rate=('matched_best', 'mean'),
).reset_index()
rates = rates.merge(pitcher_gap, on='pitcher', how='left')
display(rates[['pitcher', 'n_events', 'avoidance_rate', 'mean_rvae', 'mean_repertoire_gap', 'best_match_rate']].describe().round(4))
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && (time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py) 2>&1 | tail -10
```
Expected: `repertoire_gap 테스트 통과`, 비교 불가 건수, describe 표. `rematch_decisive.apply(..., axis=1)`은 27,391행이라 수 분 걸릴 수 있다(행 단위 반복 예측). 너무 느리면(10분 이상) 벡터화 대안(구종별로 그룹화해 배치 예측)으로 바꾸는 것을 고려하되, 우선 그대로 실행해 본다.

---

### Task 11: 사례 연구 + headless 변환/실행 + 해석

**Files:**
- Modify: `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py`
- Create: `notebooks/03_DY_run_value_gbm_pitcher_tendency.ipynb`

- [ ] **Step 1: 사례 연구 표**

Run:
```bash
cat >> $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py <<'EOF'

# %% [markdown]
# ## 11. 사례 연구
#
# 포함 기준(재대결 10건 이상)을 만족한 투수 중 회피율 상위 5명·하위 5명입니다.

# %%
name_lookup = xbh.drop_duplicates('pitcher').set_index('pitcher')['pitcher_name']
cases = rates.copy()
cases['pitcher_name'] = cases['pitcher'].map(name_lookup)
cols = ['pitcher_name', 'n_events', 'avoidance_rate', 'mean_rvae', 'mean_repertoire_gap', 'best_match_rate']
print('회피율 상위 5명 (회피형)')
display(cases.sort_values('avoidance_rate', ascending=False)[cols].head(5).round(4))
print('회피율 하위 5명 (승부형)')
display(cases.sort_values('avoidance_rate')[cols].head(5).round(4))
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py 2>&1 | tail -16
```
Expected: 두 표(상위 5명, 하위 5명)가 출력된다. 에러 없음.

- [ ] **Step 2: 해석 셀 작성 (Step 1까지의 실제 출력값을 옮겨 적는다)**

Task 4(eta², 상관계수), Task 7(MAE 개선폭), Task 9(3분위·2분위 그룹 비교의 평균·신뢰구간·p-value), Task 10(레퍼토리 일치율), Task 11 Step 1(사례)의 실제 숫자를 읽고, 파일 끝에 아래 형식으로 해석 셀을 추가한다(수치는 실제 실행 결과로 채운다 — 플레이스홀더로 남기지 않는다):

```python
# %% [markdown]
# ## 12. 해석
#
# * **시즌 안정성.** (실제 eta², 상관계수, 자격 투수 수로 채운 문장 — 안정적 특성인지 시즌마다 바뀌는지 그대로 보고)
# * **모델 성능.** GBM MAE가 베이스라인 대비 (실제 개선폭)% 낮았습니다.
# * **메인 결과.** 3분위 기준 회피형 평균 RVAE (값) vs 승부형 (값), 차이 (값) [CI], p=(값). 2분위 기준도 동일하게 보고.
#   (유의하면 어느 쪽이 유리한지, 유의하지 않으면 그렇게 명시)
# * **보조 결과.** 레퍼토리 안에서 최선과 일치한 비율이 회피형 (값) vs 승부형 (값).
# * **사례.** 상위/하위 사례 중 눈에 띄는 패턴이 있으면 1~2줄.
# * **한계.** 설계 문서 11절 그대로(결정구 Run Value가 그 전 투구 영향을 포함한 상관적 지표라는 점, 임의 임계값들, pitcher_id 미포함).
```

- [ ] **Step 3: ipynb 변환 + headless 실행**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 \
&& .venv/bin/python $NB_SRC/to_ipynb.py $NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py notebooks/03_DY_run_value_gbm_pitcher_tendency.ipynb \
&& time .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/03_DY_run_value_gbm_pitcher_tendency.ipynb
```
Expected: `wrote notebooks/03_DY_run_value_gbm_pitcher_tendency.ipynb`, `Writing ... bytes to notebooks/03_DY_run_value_gbm_pitcher_tendency.ipynb`, 에러 없음.

- [ ] **Step 4: 에러 유무·pytest 최종 확인**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && .venv/bin/python - <<'EOF'
import nbformat
nb = nbformat.read("notebooks/03_DY_run_value_gbm_pitcher_tendency.ipynb", as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
print(f"셀 {len(nb.cells)}개(코드 {len(code)}), 에러 {len(errors)}")
for e in errors:
    print("ERROR:", e.get("ename"), e.get("evalue"))
EOF
.venv/bin/python -m pytest -q 2>&1 | tail -2
git status --short
```
Expected: `에러 0`, `86 passed`, `git status`에 `requirements.txt`와 새 ipynb만 보인다(`?? notebooks/03_...ipynb`, 설계·계획 문서는 이미 있으면 그대로).

- [ ] **Step 5: 사용자에게 보고 (커밋하지 않음)**

시즌 안정성 수치, MAE 개선폭, 메인 결과(3분위·2분위 그룹 비교), 사례 연구를 요약해 보고한다. 노트북에는 실행 결과가 남아 있고, 팀 규칙대로 커밋 시 출력을 지운다는 점을 알린다. **사용자가 커밋을 요청하기 전에는 커밋하지 않는다.**

---

## 자체 검토

**스펙 커버리지**: 1절(핵심 질문)→Task 11 해석, 2절(데이터)→Task 1, 3절(회피율+안정성)→Task 3·4, 4절(GBM)→Task 6·7, 5절(RVAE)→Task 8·9, 6절(레퍼토리 반사실)→Task 10, 7절(사례)→Task 11, 8절(노트북 구조)→전체, 9절(완료 기준)→Task 11 Step 4, 11절(한계)→Task 11 Step 2.

**플레이스홀더 점검**: Task 11 Step 2의 해석 셀만 실제 수치로 채워야 하며, 형식을 명시해 뒀다. 나머지는 완전한 코드.

**타입 일관성 점검**: `season_rate`/`pitcher_rate`(Task 3), `eta_squared`(Task 4), `assign_tercile`/`assign_median`(Task 5), `build_decisive_pitch_table`/`MODEL_FEATURES`/`CATEGORICAL_FEATURES`(Task 6), `fit_run_value_model`/`final_model`(Task 7), `find_rematch_decisive_pitch`/`compute_rvae`(Task 8), `build_pitcher_repertoire`/`repertoire_gap`(Task 10) — 정의한 태스크와 쓰는 태스크에서 같은 이름·시그니처를 쓴다.

**Review Focus 반영**: 결정구 없는 재대결(Task 8 Step 3의 `missing` 로그), 레퍼토리 1개뿐(Task 10의 `repertoire_gap`이 `None` 반환 + `not_comparable` 로그), 안정성 검증 자격 투수 수 로그(Task 4), 그룹별 투수 수 로그(Task 5), 범주형 미학습 값 문제는 레퍼토리 5% 기준과 모델이 전체 표본(희귀 구종도 포함)으로 학습되는 점으로 완화됨을 Task 6에서 명시.
