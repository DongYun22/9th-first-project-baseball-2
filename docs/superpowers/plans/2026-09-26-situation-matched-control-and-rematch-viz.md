# 상황 매칭 대조군 · 재대결 구종 시각화 노트북 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 팀 research CSV만으로 (1) 상황(주자·아웃·이닝·카운트)을 맞춘 대조군 사다리 실험 노트북과 (2) 재대결 타석 구종 실행(코스·무브먼트·구속) 시각화 노트북을 만든다.

**Architecture:** 두 노트북은 각각 percent 형식 `.py` 원본(스크래치패드)으로 작성한다. 원본은 일반 스크립트로 바로 실행해 빠르게 검증하고(`assert` 셀이 테스트), 작은 변환기로 `.ipynb`를 만들어 `notebooks/`에 둔다. 저장소에는 노트북 2개, `requirements.txt` 변경, 설계·계획 문서만 남는다. 기존 `src/` 함수(`build_event_dataset_for_events`, `build_placebo_rematch`, `match_treatment_to_control`, `summarize_diff_in_diff`, `build_combined_model_dataset`, `fit_glmer`)는 소스 수정 없이 재사용한다.

**Tech Stack:** Python 3.14(`.venv`), pandas, numpy, scipy, matplotlib, R `lme4` (rpy2), Jupyter(`ipykernel`, `nbconvert`)

**설계 문서:** `docs/superpowers/specs/2026-09-25-situation-matched-control-and-rematch-viz-design.md`

## 진행 원칙 (사용자 지시)

* 데이터를 추가로 수집하지 않는다(수집기·`data/raw` 사용 금지). `_old` 폴더도 쓰지 않는다. 데이터는 팀 CSV만 쓴다.
* **커밋·push는 사용자가 요청할 때만 한다.** 그래서 이 계획에는 태스크마다 커밋 단계가 없고, 마지막 Task 12에 커밋 절차를 따로 적어 둔다. 커밋 메시지와 PR에는 Claude 관련 문구(트레일러 등)를 넣지 않는다.
* 노트북은 작업 트리에 실행 결과를 남겨 검토하고, 커밋할 때 출력을 지운다(`notebooks/README.md` 규칙).
* 검증이 실패하면(예: README 수치와 다름) 값을 맞추려고 임계값을 고치지 말고 멈추고 사용자에게 보고한다.

## 파일 구조

| 파일 | 역할 | 저장소에 남는가 |
|---|---|---|
| `notebooks/01_DY_situation_matched_control.ipynb` | 실험 1 (상황 매칭 대조군 사다리) | 예 |
| `notebooks/02_DY_rematch_pitch_execution_viz.ipynb` | 실험 2 (재대결 구종 코스·무브먼트·구속 시각화) | 예 |
| `requirements.txt` | `matplotlib`, `ipykernel`, `nbconvert` 추가 | 예 |
| `docs/superpowers/specs/…`, `docs/superpowers/plans/…` | 설계·계획 | 예 (커밋 여부는 사용자 결정) |
| `$NB_SRC/*.py`, `$NB_SRC/to_ipynb.py` | 노트북 원본과 변환기 (스크래치패드) | 아니오 |
| `data/research/*.csv`, `.venv/`, `data/processed/*` | 데이터·환경·결과물 | 아니오 (`.gitignore`) |

`NB_SRC`는 `/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src` 입니다. 셸 상태는 명령 사이에 유지되지 않으므로 아래 명령 블록마다 첫 줄에서 다시 지정합니다.

원본 파일 규칙: 셀은 `# %%`(코드) / `# %% [markdown]`(마크다운, 본문 줄은 `# `로 시작) 줄로 나눈다. 매직 명령은 `# %load_ext autoreload`처럼 주석으로 쓰고 변환기가 되살린다. `# %% 제목`처럼 `# %%` 뒤에 붙는 제목은 변환기가 무시한다.

---

### Task 0: 설계 문서 정합 수정

**Files:**
- Modify: `docs/superpowers/specs/2026-09-25-situation-matched-control-and-rematch-viz-design.md` (4.4절 5번 항목)

주자 층화를 서로 겹치지 않는 3단계로 정의한다(설계 문서의 "주자 없음 / 주자 있음 / 득점권 주자"가 겹칠 수 있는 표현이라서).

- [ ] **Step 1: 4.4절 5번 항목 수정**

`old_string`:
```
5. 주자 없음 / 주자 있음 / 득점권 주자로 나눈 층화 순수 감소
```
`new_string`:
```
5. 주자 없음 / 1루에만 주자 / 득점권 주자(2·3루 포함)로 나눈 층화 순수 감소 (서로 겹치지 않는 3단계)
```

- [ ] **Step 2: 현재 저장소 상태 확인**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && git branch --show-current && git status --short
```
Expected: `Dongyun` 과 함께 `?? docs/superpowers/plans/2026-09-26-...`, `?? docs/superpowers/specs/` 만 나온다(그 밖의 변경 없음).

---

### Task 1: 실행 환경 (venv, 패키지, CSV 복사)

**Files:**
- Modify: `requirements.txt`
- Create (git 제외): `.venv/`, `data/research/statcast_20{21..26}_*.csv`

- [ ] **Step 1: 가상환경 생성**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && python3 -m venv .venv && .venv/bin/python --version
```
Expected: `Python 3.14.x`

- [ ] **Step 2: requirements.txt에 노트북용 패키지 추가**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && python3 - <<'EOF'
from pathlib import Path

path = Path("requirements.txt")
lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
for package in ("matplotlib", "ipykernel", "nbconvert"):
    if package not in lines:
        lines.append(package)
path.write_text("\n".join(lines) + "\n")
print(path.read_text())
EOF
```
Expected: `pandas, pyarrow, pybaseball, pytest, scipy, statsmodels, rpy2, matplotlib, ipykernel, nbconvert` 가 한 줄씩 출력된다.

- [ ] **Step 3: 패키지 설치 (네트워크 사용, 소프트웨어 설치)**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && .venv/bin/pip install -r requirements.txt 2>&1 | tail -3
```
Expected: 마지막 줄이 `Successfully installed ...` (이미 있는 것은 `Requirement already satisfied`).

- [ ] **Step 4: 임포트와 R lme4 확인**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && .venv/bin/python - <<'EOF'
import pandas, pyarrow, scipy, statsmodels, matplotlib, rpy2, ipykernel, nbconvert
from src.analysis.glmer_runner import fit_glmer  # rpy2 + R 연동 확인
print("imports ok", pandas.__version__, matplotlib.__version__)
EOF
Rscript -e 'cat("lme4", as.character(packageVersion("lme4")), "\n")'
```
Expected: `imports ok <pandas 버전> <matplotlib 버전>` 과 `lme4 2.0.6`. (`Error importing in API mode ... Trying to import in ABI mode` 메시지는 무시한다.)

- [ ] **Step 5: 팀 CSV 6개 복사 (2026 파일은 이름의 ` (1)` 제거)**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && SRC=~/Downloads/drive-download-20260924T101052Z-1-001
for y in 2021 2022 2023 2024 2025; do cp "$SRC/statcast_${y}_min500_research.csv" data/research/; done
cp "$SRC/statcast_2026_min300_research (1).csv" data/research/statcast_2026_min300_research.csv
ls -l data/research | awk 'NR>1 {print $5, $9}'
```
Expected (바이트 크기, Drive 폴더의 현재 파일과 같아야 함):
```
141974435 statcast_2021_min500_research.csv
142618922 statcast_2022_min500_research.csv
144867442 statcast_2023_min500_research.csv
143240656 statcast_2024_min500_research.csv
143104961 statcast_2025_min500_research.csv
87324872 statcast_2026_min300_research.csv
```
(`README.md`가 함께 보일 수 있다.)

- [ ] **Step 6: 로더가 CSV를 인식하는지 확인**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && .venv/bin/python -c "from src.preprocessing.research_sample import list_research_csvs; print([p.name for p in list_research_csvs()])"
```
Expected: 6개 파일명이 연도순으로 출력된다(에러 없음).

- [ ] **Step 7: 기존 테스트가 통과하는지 확인**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && .venv/bin/python -m pytest -q 2>&1 | tail -3
```
Expected: `86 passed`

- [ ] **Step 8: git에 새 변경이 requirements.txt뿐인지 확인**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && git status --short
```
Expected: ` M requirements.txt`, ` M docs/superpowers/specs/...`(Task 0 수정은 spec이 untracked라 `?? docs/superpowers/specs/`로 보임), `?? docs/superpowers/plans/...` 만 있고 `.venv`, `data/` 는 나타나지 않는다.

---

### Task 2: 원본 → ipynb 변환기와 스모크 테스트

**Files:**
- Create: `$NB_SRC/to_ipynb.py`, `$NB_SRC/smoke.py`

- [ ] **Step 1: 변환기 테스트 먼저 작성**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
mkdir -p $NB_SRC && cat > $NB_SRC/test_to_ipynb.py <<'EOF'
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from to_ipynb import convert

SOURCE = '''"""preamble is ignored"""
# %% [markdown]
# # Title
#
# * item

# %% named code cell
# %load_ext autoreload
x = 1 + 1
assert x == 2

# %%
print("second")
'''

nb = convert(SOURCE)
kinds = [cell.cell_type for cell in nb.cells]
assert kinds == ["markdown", "code", "code"], kinds
assert nb.cells[0].source == "# Title\n\n* item", repr(nb.cells[0].source)
assert nb.cells[1].source.startswith("%load_ext autoreload\nx = 1 + 1"), repr(nb.cells[1].source)
assert nb.cells[2].source == 'print("second")'
assert nb.metadata["kernelspec"]["name"] == "python3"
print("to_ipynb tests passed")
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && .venv/bin/python $NB_SRC/test_to_ipynb.py
```
Expected: FAIL with `ModuleNotFoundError: No module named 'to_ipynb'`

- [ ] **Step 2: 변환기 구현**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat > $NB_SRC/to_ipynb.py <<'EOF'
"""percent 형식 .py -> .ipynb (nbformat 4).

사용: python to_ipynb.py SRC.py OUT.ipynb
"""
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

MARKER = re.compile(r"^# %%(?P<md> \[markdown\])?(?P<title>.*)$")
MAGIC = re.compile(r"^# (%[A-Za-z].*)$")  # "# %load_ext autoreload" -> "%load_ext autoreload"


def convert(source_text: str):
    cells, kind, buffer = [], None, []

    def flush():
        body = "\n".join(buffer).strip("\n")
        if kind is None or not body.strip():
            return
        if kind == "markdown":
            cells.append(new_markdown_cell("\n".join(re.sub(r"^# ?", "", line) for line in body.split("\n"))))
        else:
            cells.append(new_code_cell("\n".join(MAGIC.sub(r"\1", line) for line in body.split("\n"))))

    for line in source_text.split("\n"):
        match = MARKER.match(line)
        if match:
            flush()
            kind = "markdown" if match.group("md") else "code"
            buffer = []
        else:
            buffer.append(line)
    flush()

    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    return notebook


if __name__ == "__main__":
    source_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    nbformat.write(convert(source_path.read_text(encoding="utf-8")), out_path)
    print(f"wrote {out_path}")
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && .venv/bin/python $NB_SRC/test_to_ipynb.py
```
Expected: `to_ipynb tests passed`

- [ ] **Step 3: 변환 + headless 실행 스모크**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat > $NB_SRC/smoke.py <<'EOF'
# %% [markdown]
# # smoke
# * item

# %%
# %load_ext autoreload
# %autoreload 2
x = 1 + 1
assert x == 2
print("smoke ok", x)
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && .venv/bin/python $NB_SRC/to_ipynb.py $NB_SRC/smoke.py $NB_SRC/smoke.ipynb \
&& .venv/bin/jupyter nbconvert --to notebook --execute $NB_SRC/smoke.ipynb --output smoke_out.ipynb --output-dir $NB_SRC 2>&1 | tail -2 \
&& grep -c "smoke ok 2" $NB_SRC/smoke_out.ipynb
```
Expected: `Writing ... bytes to .../smoke_out.ipynb` 와 함께 마지막 줄 `1` (또는 그 이상). 커널을 못 찾는다는 에러(`No such kernel named python3`)가 나오면 `.venv/bin/python -m ipykernel install --user --name python3` 후 다시 실행한다.

---

### Task 3: 노트북 1 — 설정, CSV 읽기, 상황 변수

**Files:**
- Create: `$NB_SRC/01_DY_situation_matched_control.py`

- [ ] **Step 1: 원본 파일 생성 (제목, 설정, 임포트, CSV 읽기, 표본 검증)**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat > $NB_SRC/01_DY_situation_matched_control.py <<'EOF'
# %% [markdown]
# # ⚾ 상황 매칭 대조군 실험: 주자·아웃·이닝·카운트를 맞춰도 장타 후 구종 회피가 남는가?
#
# * **작성자:** DY
# * **작성일:** 2026-09-26
# * **목적:**
#   * README의 한계("대조군이 투수·이닝·카운트·주자 상황까지 매칭된 것은 아님 — 장타는 위기 상황과 함께 나오는 경우가 많아 '장타를 맞아서'와 '위기라서'가 섞였을 가능성")를 실험합니다.
#   * 팀 공유 research CSV의 `on_1b/on_2b/on_3b`(투구 직전 주자)로 대조군을 상황별로 다시 뽑고(M0 → M1 → M2), 순수 감소와 혼합모델 `group` 오즈비가 어떻게 변하는지 봅니다.
# * **설계 문서:** `docs/superpowers/specs/2026-09-25-situation-matched-control-and-rematch-viz-design.md`
# * **주의:** CSV에는 점수가 없어 점수차는 매칭과 모델에서 모두 뺐습니다. 점수차를 통제하지 못한 것이 이 실험의 한계입니다.

# %%
# 1. 경로 설정 (src 폴더 접근용)
import sys, os
sys.path.append(os.path.abspath('..'))

# 2. Autoreload 설정 (src 내부의 py 파일 수정 시 즉시 반영)
# %load_ext autoreload
# %autoreload 2

# 경고 메시지 무시 (선택 사항)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from IPython.display import display

from src.analysis.avoidance_stats import match_treatment_to_control, summarize_diff_in_diff
from src.analysis.glmer_runner import check_convergence, extract_fixed_effects_table, fit_glmer
from src.analysis.mixed_effects_model import build_combined_model_dataset
from src.analysis.run_same_batter_rematch_analysis import PLACEBO_SEED, build_placebo_rematch, build_xbh_rematch
from src.analysis.avoidance_stats import compute_season_usage_rate
from src.preprocessing.build_next_ab_dataset import identify_extra_base_hit_events
from src.preprocessing.research_sample import KEY_COLUMNS, list_research_csvs

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

# %% [markdown]
# ## 1. 팀 CSV 읽기
#
# `data/raw`를 쓰지 않고 팀 research CSV 6개를 그대로 읽습니다. CSV에는 점수(`home_score`, `away_score`)가 없어서,
# 기존 이벤트 빌더가 요구하는 점수 컬럼을 0으로 채워 호출하고 결과의 `score_diff`는 바로 버립니다(어디에도 쓰지 않습니다).

# %%
LOAD_COLUMNS = [
    'game_pk', 'game_date', 'at_bat_number', 'pitch_number', 'pitcher', 'player_name', 'batter',
    'stand', 'p_throws', 'pitch_type', 'events', 'balls', 'strikes', 'outs_when_up',
    'inning', 'inning_topbot', 'on_1b', 'on_2b', 'on_3b',
]


def load_team_csv() -> pd.DataFrame:
    csv_files = list_research_csvs()  # data/research/ (형식이 다른 파일명이 있으면 에러)
    frames = [pd.read_csv(path, usecols=LOAD_COLUMNS, encoding='utf-8-sig', low_memory=False) for path in csv_files]
    pitches = pd.concat(frames, ignore_index=True)
    pitches['season'] = pd.to_datetime(pitches['game_date']).dt.year
    pitches['home_score'] = 0  # CSV에 점수 없음: 빌더 호출용 자리 채움 (score_diff는 곧 버림)
    pitches['away_score'] = 0
    return pitches


pitches = load_team_csv()
print(f'{len(pitches):,}행, 투수 {pitches["pitcher"].nunique():,}명, 시즌 {sorted(pitches["season"].unique())}')

# %%
assert len(pitches) == 3_592_302, len(pitches)
assert pitches['pitcher'].nunique() == 1_067
assert not pitches.duplicated(KEY_COLUMNS).any(), 'CSV 키(game_pk, at_bat_number, pitch_number, pitcher)가 중복'
assert pitches['outs_when_up'].isin([0, 1, 2]).all()
assert pitches[['inning', 'balls', 'strikes']].notna().all().all()
print('표본 검증 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -5
```
Expected: `3,592,302행, 투수 1,067명, 시즌 [2021, 2022, 2023, 2024, 2025, 2026]` 와 `표본 검증 통과`

- [ ] **Step 2: 상황 변수 테스트 셀을 먼저 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% test: situation columns
_t = pd.DataFrame({
    'on_1b': [np.nan, 1.0, np.nan, 5.0], 'on_2b': [np.nan, np.nan, 2.0, 6.0], 'on_3b': [np.nan, np.nan, np.nan, 7.0],
    'outs_when_up': [0, 1, 2, 2], 'inning': [1, 4, 9, 7], 'balls': [0, 2, 1, 3], 'strikes': [0, 1, 2, 1],
})
_r = add_situation_columns(_t)
assert _r['runners_n'].tolist() == [0, 1, 1, 3]
assert _r['risp'].tolist() == [0, 0, 1, 1]
assert _r['bases'].tolist() == [0, 1, 2, 7]
assert _r['base_out_state'].tolist() == [0, 4, 8, 23]
assert _r['inning_bucket'].tolist() == ['1-3', '4-6', '7+', '7+']
assert _r['count_bucket'].tolist() == ['even', 'batter', 'pitcher', 'batter']
print('add_situation_columns 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: FAIL with `NameError: name 'add_situation_columns' is not defined`

- [ ] **Step 3: 구현 셀을 테스트 셀 바로 앞에 삽입**

Edit tool로 `$NB_SRC/01_DY_situation_matched_control.py`에서 아래를 수행한다.

`old_string`:
```
# %% test: situation columns
```
`new_string`:
```
# %% [markdown]
# ## 2. 상황 변수
#
# 이벤트를 만든 투구 시점(투구 직전)의 주자·아웃·이닝·카운트로 매칭용 변수를 만듭니다.
# `bases`는 1루=1, 2루=2, 3루=4를 더한 0–7, `base_out_state`는 `bases × 3 + 아웃`(0–23)입니다.

# %% situation columns
SITUATION_COLUMNS = ['runners_n', 'risp', 'bases', 'base_out_state', 'inning_bucket', 'count_bucket']


def add_situation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    on1, on2, on3 = out['on_1b'].notna(), out['on_2b'].notna(), out['on_3b'].notna()
    out['runners_n'] = on1.astype(int) + on2.astype(int) + on3.astype(int)
    out['risp'] = (on2 | on3).astype(int)
    out['bases'] = on1.astype(int) + 2 * on2.astype(int) + 4 * on3.astype(int)
    out['base_out_state'] = out['bases'] * 3 + out['outs_when_up'].astype(int)
    out['inning_bucket'] = np.select([out['inning'] <= 3, out['inning'] <= 6], ['1-3', '4-6'], default='7+')
    out['count_bucket'] = np.select(
        [out['strikes'] > out['balls'], out['strikes'] < out['balls']], ['pitcher', 'batter'], default='even'
    )
    return out


# %% test: situation columns
```

- [ ] **Step 4: 실제 데이터에 적용하는 셀을 파일 끝에 추가하고 실행**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %%
pitches = add_situation_columns(pitches)
print(pitches[['runners_n', 'risp', 'base_out_state']].describe().round(3))
print(pitches['inning_bucket'].value_counts(normalize=True).round(3).to_dict())
print(pitches['count_bucket'].value_counts(normalize=True).round(3).to_dict())
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -16
```
Expected: `add_situation_columns 테스트 통과` 이후 `runners_n` 평균이 약 0.6 안팎(0.5–0.7), `risp` 평균이 약 0.2–0.3, `base_out_state` 최대값 23, `inning_bucket`과 `count_bucket` 비율이 각각 합 1로 출력된다. 에러가 없어야 한다.

---

### Task 4: 노트북 1 — 이벤트 만들기와 환경 검증, 상황 붙이기

**Files:**
- Modify: `$NB_SRC/01_DY_situation_matched_control.py`

- [ ] **Step 1: 장타 재대결 이벤트 생성 + 건수 검증 셀 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% [markdown]
# ## 3. 장타 재대결 이벤트와 환경 검증
#
# 기존 빌더로 같은 타자 재대결 이벤트를 만듭니다. 건수는 점수와 무관하므로 README와 정확히 같아야 합니다.

# %%
season_usage = compute_season_usage_rate(pitches)
xbh_all = identify_extra_base_hit_events(pitches)
assert len(xbh_all) == 70_319, len(xbh_all)

xbh = build_xbh_rematch(pitches, season_usage).drop(columns='score_diff')
n_eligible = int(xbh['pitch_family'].notna().sum())
assert len(xbh) == 27_391, len(xbh)
assert n_eligible == 27_371, n_eligible
print(f'장타 {len(xbh_all):,}건, 같은 타자 재대결 {len(xbh):,}건 (구종 있는 {n_eligible:,}건) -- README와 일치')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -6
```
Expected: 마지막에 `장타 70,319건, 같은 타자 재대결 27,391건 (구종 있는 27,371건) -- README와 일치`. 소요 시간은 수 분. 어느 `assert`든 실패하면 **멈추고 실제 값을 사용자에게 보고한다**.

- [ ] **Step 2: 상황 붙이기 테스트 셀 먼저 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% test: attach situation
_ev = pd.DataFrame({
    'game_pk': [1, 1], 'pitcher': [10, 10], 'at_bat_number': [5, 9],
    'event_pitch_number': [3, 2], 'next_at_bat_number': [9.0, 14.0],
})
_px = pd.DataFrame({
    'game_pk': [1] * 5, 'pitcher': [10] * 5, 'at_bat_number': [5, 5, 9, 9, 14], 'pitch_number': [1, 3, 1, 2, 4],
    'runners_n': [0, 2, 1, 1, 3], 'risp': [0, 1, 0, 0, 1], 'bases': [0, 5, 1, 1, 7],
    'base_out_state': [0, 15, 3, 3, 23], 'inning_bucket': ['1-3'] * 5, 'count_bucket': ['even'] * 5,
    'outs_when_up': [0, 0, 1, 1, 2],
})
_a = attach_event_situation(_ev, _px)
assert _a['runners_n'].tolist() == [2, 1] and _a['risp'].tolist() == [1, 0]
_b = attach_next_situation(_a, _px)
assert _b['runners_n_next'].tolist() == [1, 3]
assert _b['risp_next'].tolist() == [0, 1]
assert _b['outs_next'].tolist() == [1, 2]
assert len(_b) == 2
print('attach_*_situation 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: FAIL with `NameError: name 'attach_event_situation' is not defined`

- [ ] **Step 3: 구현 셀을 테스트 셀 바로 앞에 삽입**

Edit tool로 `$NB_SRC/01_DY_situation_matched_control.py`에서:

`old_string`:
```
# %% test: attach situation
```
`new_string`:
```
# %% attach situation
PITCH_KEYS = ['game_pk', 'at_bat_number', 'pitch_number', 'pitcher']
NEXT_COLUMNS = ['runners_n_next', 'risp_next', 'outs_next']


def attach_event_situation(events: pd.DataFrame, pitches_sit: pd.DataFrame) -> pd.DataFrame:
    """이벤트를 만든 투구의 상황 변수를 이벤트 표에 붙인다 (장타·대조군 공통)."""
    events = events.drop(columns=[c for c in SITUATION_COLUMNS if c in events.columns])
    situation = pitches_sit[PITCH_KEYS + SITUATION_COLUMNS].rename(columns={'pitch_number': 'event_pitch_number'})
    out = events.merge(
        situation, on=['game_pk', 'at_bat_number', 'event_pitch_number', 'pitcher'], how='left', validate='many_to_one'
    )
    assert out[SITUATION_COLUMNS].notna().all().all(), '이벤트 투구를 CSV에서 찾지 못함'
    return out


def attach_next_situation(events: pd.DataFrame, pitches_sit: pd.DataFrame) -> pd.DataFrame:
    """재대결 타석에서 같은 투수가 던진 첫 투구 시점의 주자·아웃을 붙인다 (균형 점검·보정용)."""
    events = events.drop(columns=[c for c in NEXT_COLUMNS if c in events.columns])
    first = (
        pitches_sit.sort_values(['game_pk', 'at_bat_number', 'pitcher', 'pitch_number'])
        .groupby(['game_pk', 'at_bat_number', 'pitcher'], as_index=False)
        .first()[['game_pk', 'at_bat_number', 'pitcher', 'runners_n', 'risp', 'outs_when_up']]
        .rename(columns={
            'at_bat_number': 'next_at_bat_number', 'runners_n': 'runners_n_next',
            'risp': 'risp_next', 'outs_when_up': 'outs_next',
        })
    )
    first['next_at_bat_number'] = first['next_at_bat_number'].astype(float)
    return events.merge(first, on=['game_pk', 'next_at_bat_number', 'pitcher'], how='left', validate='many_to_one')


# %% test: attach situation
```

- [ ] **Step 4: 장타 이벤트에 붙이는 셀을 파일 끝에 추가하고 실행**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %%
xbh = attach_next_situation(attach_event_situation(xbh, pitches), pitches)
assert len(xbh) == 27_391
display(xbh[['runners_n', 'risp', 'outs_when_up', 'runners_n_next', 'risp_next']].mean().round(3).to_frame('장타 평균'))
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -9
```
Expected: `attach_*_situation 테스트 통과` 이후 장타 평균 표가 나온다. `runners_n`은 0.4–0.6(이벤트 시점), `runners_n_next`는 약 0.5–0.7 범위, 에러 없음.

---

### Task 5: 노트북 1 — 균형·순수 감소 도우미와 사다리 빌더

**Files:**
- Modify: `$NB_SRC/01_DY_situation_matched_control.py`

- [ ] **Step 1: SMD·균형표·순수 감소 테스트 셀 먼저 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% test: balance and reduction helpers
assert np.isclose(smd(pd.Series([0.0, 2.0]), pd.Series([1.0, 3.0])), -1 / np.sqrt(2))
assert smd(pd.Series([1.0, 1.0]), pd.Series([1.0, 1.0])) == 0.0
_bt = balance_table(pd.DataFrame({'a': [0.0, 2.0]}), pd.DataFrame({'a': [1.0, 3.0]}), ['a'])
assert list(_bt.columns) == ['xbh_mean', 'placebo_mean', 'smd'] and np.isclose(_bt.loc['a', 'smd'], -1 / np.sqrt(2))

_x = pd.DataFrame({'has_next_ab': [True, True, False], 'b': [0.5, 0.5, 0.5], 'same_type_share': [0.3, 0.1, 0.0]})
_p = pd.DataFrame({'has_next_ab': [True, True], 'b': [0.5, 0.5], 'same_type_share': [0.5, 0.3]})
_r = pure_reduction(_x, _p, 'b')
assert _r['n_xbh'] == 2 and _r['n_placebo'] == 2 and np.isclose(_r['net'], 0.2), _r

_g = runner_group(pd.DataFrame({'risp': [0, 0, 1, 1], 'runners_n': [0, 1, 1, 2]}))
assert _g.tolist() == ['주자 없음', '1루만', '득점권', '득점권']
print('균형·순수 감소 도우미 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: FAIL with `NameError: name 'smd' is not defined`

- [ ] **Step 2: 구현 셀을 테스트 셀 바로 앞에 삽입**

Edit tool로 `$NB_SRC/01_DY_situation_matched_control.py`에서:

`old_string`:
```
# %% test: balance and reduction helpers
```
`new_string`:
```
# %% [markdown]
# ## 4. 도우미: 균형표(SMD), 순수 감소, 주자 층
#
# * `smd`: 표준화 평균차(풀링 표준편차). |SMD| < 0.1이면 균형이 맞다고 봅니다.
# * `pure_reduction`: `diff = baseline − same_type_share`의 장타 평균 − 대조군 평균 (기존 분석과 같은 정의)
# * `runner_group`: 서로 겹치지 않는 3단계 (주자 없음 / 1루만 / 득점권)

# %% balance and reduction helpers
def smd(x: pd.Series, y: pd.Series) -> float:
    x, y = x.dropna(), y.dropna()
    pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    diff = x.mean() - y.mean()
    if pooled == 0:
        return 0.0 if diff == 0 else float('inf')
    return diff / pooled


def balance_table(xbh_: pd.DataFrame, placebo_: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = {
        c: {'xbh_mean': xbh_[c].mean(), 'placebo_mean': placebo_[c].mean(), 'smd': smd(xbh_[c], placebo_[c])}
        for c in columns
    }
    return pd.DataFrame.from_dict(rows, orient='index')


def usable(events: pd.DataFrame, baseline_col: str) -> pd.DataFrame:
    d = events[events['has_next_ab'] & events[baseline_col].notna()].copy()
    d['diff'] = d[baseline_col] - d['same_type_share']
    return d


def pure_reduction(xbh_: pd.DataFrame, placebo_: pd.DataFrame, baseline_col: str) -> dict:
    x, p = usable(xbh_, baseline_col), usable(placebo_, baseline_col)
    r = summarize_diff_in_diff(x['diff'], p['diff'])
    return {
        'n_xbh': r['n_treatment'], 'n_placebo': r['n_control'], 'net': r['net_effect'],
        'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
    }


def runner_group(events: pd.DataFrame) -> np.ndarray:
    return np.select([events['risp'] == 1, events['runners_n'] >= 1], ['득점권', '1루만'], default='주자 없음')


# %% test: balance and reduction helpers
```

- [ ] **Step 3: 도우미 테스트 통과 확인**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: `균형·순수 감소 도우미 테스트 통과`

- [ ] **Step 4: 사다리 정의와 빌더를 파일 끝에 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% [markdown]
# ## 5. 매칭 사다리 (M0 → M1 → M2)
#
# | 단계 | 대조군을 맞추는 칸 |
# |---|---|
# | M0 | 시즌 × 구종계열 (현행) |
# | M1 | M0 + 주자·아웃 24가지 (`base_out_state`) |
# | M2 | M1 + 이닝 구간 + 카운트 구간 |
#
# 대조군은 기존 `build_placebo_rematch`(칸 지정 가능)로 뽑고, `match_treatment_to_control`로 장타를 대조군 수에 맞춰
# 칸마다 양쪽 크기를 같게 합니다. 점수차는 CSV에 없어 칸에 넣지 못했습니다.

# %%
STEPS = {
    'M0': ('season', 'pitch_family'),
    'M1': ('season', 'pitch_family', 'base_out_state'),
    'M2': ('season', 'pitch_family', 'base_out_state', 'inning_bucket', 'count_bucket'),
}


def build_step(step: str, pitches_sit: pd.DataFrame, xbh_: pd.DataFrame, season_usage_: pd.DataFrame):
    """단계별 대조군을 뽑고 장타를 칸별 대조군 수에 맞춘다. (장타, 대조군) 반환."""
    strata = list(STEPS[step])
    placebo = build_placebo_rematch(pitches_sit, xbh_, season_usage_, strata_cols=tuple(strata)).drop(columns='score_diff')
    placebo = attach_next_situation(attach_event_situation(placebo, pitches_sit), pitches_sit)
    xbh_m = match_treatment_to_control(xbh_, placebo, tuple(strata), seed=PLACEBO_SEED)
    same = xbh_m.groupby(strata).size().sort_index().equals(placebo.groupby(strata).size().sort_index())
    assert same, f'{step}: 칸별 표본 크기가 양쪽에서 다름'
    return xbh_m, placebo


pairs = {}
for step in STEPS:
    xbh_m, placebo_m = build_step(step, pitches, xbh, season_usage)
    pairs[step] = (xbh_m, placebo_m)
    print(f'{step}: 장타 {len(xbh_m):,}/{n_eligible:,}건 유지 ({len(xbh_m) / n_eligible:.1%}), 대조군 {len(placebo_m):,}건')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -6
```
Expected: `M0: 장타 27,371/27,371건 유지 (100.0%), 대조군 27,371건`, 이어서 `M1`, `M2` 줄이 나온다(M1·M2의 유지율은 결과로 확인). 소요 시간은 10분 안팎일 수 있으니 백그라운드로 실행해도 된다. 실패하면 에러 메시지를 그대로 보고 원인을 찾는다(추측으로 고치지 않는다).

- [ ] **Step 5: M2 유지율 점검**

Step 4 출력의 `M2` 유지율이 90% 이상이면 이 스텝은 건너뛴다. **90% 미만이면** 설계 문서 4.3절 규칙에 따라 구간을 더 거칠게 합치고 그 사실을 노트북에 적는다. 이 경우 파일 끝에 아래 셀을 추가하고, Step 4의 `STEPS['M2']`를 `('season', 'pitch_family', 'coarse_state', 'inning_bucket2', 'count_bucket2')`로 바꿔 `pitches = add_coarse_situation(pitches)` 를 `pitches = add_situation_columns(pitches)` 바로 다음 셀에 둔 뒤 다시 실행한다(`attach_*` 함수의 `SITUATION_COLUMNS`에 새 컬럼도 추가).

```python
# %% coarse situation (M2 유지율 90% 미만일 때만 사용)
def add_coarse_situation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['runners_bucket'] = np.minimum(out['runners_n'], 2)                 # 0, 1, 2+
    out['coarse_state'] = out['runners_bucket'] * 3 + out['outs_when_up'].astype(int)   # 9가지
    out['inning_bucket2'] = np.where(out['inning'] <= 4, '1-4', '5+')
    out['count_bucket2'] = np.where(out['strikes'] > out['balls'], 'pitcher', 'other')
    return out
```
`SITUATION_COLUMNS`에는 `'coarse_state', 'inning_bucket2', 'count_bucket2'`를 추가한다. 90% 이상이었다면 이 셀은 만들지 않는다.

---

### Task 6: 노트북 1 — M0 환경 검증과 균형표

**Files:**
- Modify: `$NB_SRC/01_DY_situation_matched_control.py`

- [ ] **Step 1: M0가 README를 재현하는지 검증하는 셀 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% [markdown]
# ## 6. 환경 검증: M0가 README 수치를 재현하는가
#
# 장타 쪽 값은 결정적이라 정확히 같아야 하고, 대조군은 무작위 추출이라 행 순서에 따라 조금 달라질 수 있어
# 순수 감소는 README 신뢰구간 안이면 통과로 봅니다. (점수차가 없으므로 혼합모델 오즈비 0.286은 비교 대상이 아닙니다.)

# %%
xbh0, placebo0 = pairs['M0']
assert len(xbh0) / n_eligible >= 0.999, 'M0에서 장타가 줄어듦: 대조군 후보가 부족한 칸이 있음'

season_m0 = pure_reduction(xbh0, placebo0, 'season_usage_rate')
game_m0 = pure_reduction(xbh0, placebo0, 'baseline_usage')
reuse_x = usable(xbh0, 'season_usage_rate')['reused_same_type'].mean()
reuse_p = usable(placebo0, 'season_usage_rate')['reused_same_type'].mean()
print(f'시즌 baseline 순수 감소 {season_m0["net"]:.4f} [{season_m0["ci_low"]:.4f}, {season_m0["ci_high"]:.4f}]  (README 0.146 [0.142, 0.151])')
print(f'경기 내 baseline 순수 감소 {game_m0["net"]:.4f} [{game_m0["ci_low"]:.4f}, {game_m0["ci_high"]:.4f}]  (README 0.156 [0.151, 0.161])')
print(f'재사용률 장타 {reuse_x:.3f} vs 대조군 {reuse_p:.3f}  (README 0.455 vs 0.679)')

assert 0.142 <= season_m0['net'] <= 0.151, season_m0
assert 0.151 <= game_m0['net'] <= 0.161, game_m0
assert abs(reuse_x - 0.455) < 0.002, reuse_x
assert abs(reuse_p - 0.679) < 0.010, reuse_p
print('M0 환경 검증 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -6
```
Expected: 세 줄의 수치 출력 후 `M0 환경 검증 통과`. `assert`가 실패하면 **멈추고 실제 값과 README 값을 사용자에게 보고한다**(임계값을 바꾸지 않는다).

- [ ] **Step 2: 균형표 셀 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% [markdown]
# ## 7. 균형표 (표준화 평균차, SMD)
#
# 이벤트 시점 변수와 재대결 시점 변수(`*_next`)를 단계별로 비교합니다. 점수차는 데이터에 없어 점검하지 못합니다.
# 매칭에 쓴 변수(주자·아웃 등)는 M1·M2에서 0에 가까워지는 것이 정상이고, 매칭에 쓰지 않은 변수가 어떻게 움직이는지가 볼거리입니다.

# %%
BALANCE_COLUMNS = [
    'runners_n', 'risp', 'outs_when_up', 'inning', 'balls', 'strikes', 'platoon_match', 'runners_n_next', 'risp_next',
]
print('M0 평균 비교')
display(balance_table(*pairs['M0'], BALANCE_COLUMNS).round(3))

balance_df = pd.DataFrame({step: balance_table(*pairs[step], BALANCE_COLUMNS)['smd'] for step in STEPS}).round(3)
print('단계별 SMD')
display(balance_df)
print('|SMD| >= 0.1 인 항목:', {s: balance_df.index[balance_df[s].abs() >= 0.1].tolist() for s in STEPS})
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -30
```
Expected: M0 평균 표와 단계별 SMD 표가 나온다. `runners_n`, `risp`의 SMD는 M0에서 |0.1| 안팎, M1·M2에서 0에 가까워야 한다. 마지막 줄에 단계별 `|SMD| >= 0.1` 항목이 출력된다.

---

### Task 7: 노트북 1 — 순수 감소, 혼합모델, 층화, 요약

**Files:**
- Modify: `$NB_SRC/01_DY_situation_matched_control.py`

- [ ] **Step 1: 공식 테스트 셀 먼저 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% test: formulas
assert 'score_diff' not in BASE_FORMULA and 'score_diff' not in C0_FORMULA
assert C0_FORMULA.count('runners_n + risp') == 1 and 'runners_n' not in BASE_FORMULA
assert C0_FORMULA.endswith('(0 + group | pitcher)') and BASE_FORMULA.endswith('(0 + group | pitcher)')
print('공식 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: FAIL with `NameError: name 'BASE_FORMULA' is not defined`

- [ ] **Step 2: 구현 셀을 테스트 셀 바로 앞에 삽입**

Edit tool로 `$NB_SRC/01_DY_situation_matched_control.py`에서:

`old_string`:
```
# %% test: formulas
```
`new_string`:
```
# %% [markdown]
# ## 8. 결과: 순수 감소와 혼합모델
#
# * 혼합모델 공식은 README 공식에서 `score_diff`만 뺀 것을 모든 단계에 똑같이 적용합니다. 그래서 M0의 `group` 오즈비는 README의 0.286과 정확히 같지 않을 수 있고, 비교는 단계 사이에서만 합니다.
# * C0는 M0 표본에 `runners_n`, `risp`를 공변량으로 더한 교차검증입니다(대조군은 그대로).

# %% mixed model helpers
BASE_FORMULA = (
    'reused_same_type ~ group * baseline_usage + balls + strikes + outs_when_up '
    '+ stand + pitch_family + factor(season) + (0 + group | pitcher)'
)
C0_FORMULA = BASE_FORMULA.replace(' + (0 + group | pitcher)', ' + runners_n + risp + (0 + group | pitcher)')
REQUIRED = [
    'reused_same_type', 'baseline_usage', 'balls', 'strikes', 'outs_when_up', 'stand', 'pitch_family', 'season', 'pitcher',
]


def fit_mixed(xbh_: pd.DataFrame, placebo_: pd.DataFrame, formula: str, extra_required=()) -> dict:
    xf = xbh_[xbh_['has_next_ab'] & xbh_['baseline_usage'].notna()]
    pf = placebo_[placebo_['has_next_ab'] & placebo_['baseline_usage'].notna()]
    combined = build_combined_model_dataset(xf, pf, required_columns=REQUIRED + list(extra_required))
    fit_glmer(combined, formula)
    return {
        'n': len(combined), 'convergence': check_convergence(),
        'fe': extract_fixed_effects_table().set_index('term'),
    }


def key_terms(fit: dict) -> dict:
    fe = fit['fe']
    return {
        'group_or': fe.loc['group', 'odds_ratio'],
        'group_or_low': fe.loc['group', 'or_ci_low'],
        'group_or_high': fe.loc['group', 'or_ci_high'],
        'inter_or': fe.loc['group:baseline_usage', 'odds_ratio'],
        'inter_p': fe.loc['group:baseline_usage', 'p_value'],
    }


# %% test: formulas
```

- [ ] **Step 3: 공식 테스트 통과 확인**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: `공식 테스트 통과`

- [ ] **Step 4: 사다리 요약표 셀 추가 (M0, M1, M2, C0)**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% ladder summary
fits, rows = {}, []
for step, (xbh_m, placebo_m) in pairs.items():
    fits[step] = fit_mixed(xbh_m, placebo_m, BASE_FORMULA)
    season, game = pure_reduction(xbh_m, placebo_m, 'season_usage_rate'), pure_reduction(xbh_m, placebo_m, 'baseline_usage')
    rows.append({
        'step': step, 'n_xbh': len(xbh_m), 'retention': len(xbh_m) / n_eligible, 'n_model': fits[step]['n'],
        'net_season': season['net'], 'season_low': season['ci_low'], 'season_high': season['ci_high'],
        'net_game': game['net'], 'game_low': game['ci_low'], 'game_high': game['ci_high'],
        **key_terms(fits[step]),
    })

# C0: M0 표본 + 주자 공변량 (순수 감소는 M0와 같은 표본이므로 M0 값을 그대로 쓴다)
fits['C0'] = fit_mixed(*pairs['M0'], C0_FORMULA, extra_required=('runners_n', 'risp'))
rows.append({**rows[0], 'step': 'C0', 'n_model': fits['C0']['n'], **key_terms(fits['C0'])})

summary = pd.DataFrame(rows).set_index('step')
display(summary.round(4))
print({step: fit['convergence'] for step, fit in fits.items()})
summary.to_csv('../data/processed/situation_matched_ladder_summary.csv')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -14
```
Expected: `M0/M1/M2/C0` 네 줄짜리 요약표(유지율, 순수 감소와 CI, `group_or`, `inter_or` 포함)와 단계별 수렴 메시지 딕셔너리가 나온다. `group_or`는 1보다 작고(약 0.2–0.4 범위), `inter_or`는 1보다 크며(약 1.5–2.5), 수렴 메시지에 `isSingular=False`가 보이면 정상이다. 에러가 나면 원인을 확인한다(예: 컬럼 누락).

- [ ] **Step 5: 주자별 층화 테스트 없이 바로 구현 (기존 검증된 함수 조합)**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/01_DY_situation_matched_control.py <<'EOF'

# %% [markdown]
# ## 9. 주자 상황별 층화 순수 감소
#
# 같은 주자 상황 안에서도 장타와 대조군의 차이가 남는지 봅니다. 3단계는 서로 겹치지 않습니다:
# 주자 없음 / 1루에만 주자 / 득점권(2·3루 주자, 1루 동시 포함). 각 칸에 양쪽 30건 미만이면 건너뜁니다.

# %%
def stratified_reduction(xbh_: pd.DataFrame, placebo_: pd.DataFrame, baseline_col: str, min_n: int = 30) -> pd.DataFrame:
    x, p = usable(xbh_, baseline_col), usable(placebo_, baseline_col)
    x['runner_group'], p['runner_group'] = runner_group(x), runner_group(p)
    rows = []
    for level in ['주자 없음', '1루만', '득점권']:
        gx, gp = x[x['runner_group'] == level], p[p['runner_group'] == level]
        if len(gx) < min_n or len(gp) < min_n:
            continue
        r = summarize_diff_in_diff(gx['diff'], gp['diff'])
        rows.append({
            'runner_group': level, 'n_xbh': len(gx), 'n_placebo': len(gp),
            'net': r['net_effect'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
        })
    return pd.DataFrame(rows)


strat = pd.concat(
    [
        stratified_reduction(*pairs[step], baseline).assign(step=step, baseline=baseline)
        for step in STEPS
        for baseline in ('season_usage_rate', 'baseline_usage')
    ],
    ignore_index=True,
)
display(strat.pivot_table(index=['baseline', 'runner_group'], columns='step', values='net').round(4))
display(strat.round(4))
strat.to_csv('../data/processed/situation_matched_stratified_by_runners.csv', index=False)
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/01_DY_situation_matched_control.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -30
```
Expected: 요약 피벗표(행: baseline × 주자 3단계, 열: M0/M1/M2)와 상세표가 출력된다. 값은 양수의 %p 단위(예: 0.10–0.20 범위)이고 에러가 없다.

- [ ] **Step 6: 해석 셀 작성 (결과를 본 뒤)**

Step 4·5의 표를 읽고, 설계 문서 4.5절의 사전 정의 원칙으로 마크다운 해석 셀을 파일 끝에 추가한다. **숫자는 실제 출력값을 그대로 옮긴다.** 문안 규칙:

* M0 → M2의 `net_season`, `net_game` 변화가 M0 신뢰구간 폭 안이면: "관측한 주자·아웃·이닝·카운트를 맞춰도 회피가 남는다. 순수 감소는 M0 X%p → M2 Y%p."
* 눈에 띄게 줄면(M0 신뢰구간 밖): "일부 혼입. 줄어든 크기는 X%p → Y%p."
* 항상 덧붙일 한계 3가지: 점수차를 통제하지 못함, 레버리지·투수 피로 등 미관측 요인 배제 불가, 주자는 투구 직전 상태.
* 주자 층별 결과가 세 층 모두 양수이면 그 사실도 한 줄로 적는다.
* 유지율(M1, M2)과 `|SMD| ≥ 0.1`로 남은 항목을 함께 적는다.

셀 형식:
```python
# %% [markdown]
# ## 10. 해석 (사전 정의 원칙: 설계 문서 4.5절)
#
# * (실제 수치로 채운 해석 문장들)
```
(위 규칙에 따라 실제 수치로 문장을 만든다. 수치가 없는 문장을 남기지 않는다.)

---

### Task 8: 노트북 2 — 읽기, 이벤트, 같은 구종 투구 수집

**Files:**
- Create: `$NB_SRC/02_DY_rematch_pitch_execution_viz.py`

- [ ] **Step 1: 원본 파일 생성 (제목, 설정, CSV 읽기, 이벤트 생성 및 검증)**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat > $NB_SRC/02_DY_rematch_pitch_execution_viz.py <<'EOF'
# %% [markdown]
# # ⚾ 재대결에서 같은 구종은 어떻게 달라졌나: 코스·무브먼트·구속 시각화
#
# * **작성자:** DY
# * **작성일:** 2026-09-26
# * **목적:**
#   * 장타를 허용한 구종을 같은 타자와의 재대결 타석에서 다시 던졌을 때, 직전(장타) 타석과 비교해 같은 구종의 코스·무브먼트·구속이 어떻게 달라졌는지 봅니다.
#   * 투수 손(우투/좌투)별로 그림을 나누고(같은 구종이라도 손에 따라 궤적이 다름), 장타 허용 타석과 재대결 타석을 색으로, 구종별로 행을 나눠 보여 줍니다.
# * **설계 문서:** `docs/superpowers/specs/2026-09-25-situation-matched-control-and-rematch-viz-design.md`
# * **한계:** 대조군이 없어 변화가 "장타 반응"이라고 단정할 수 없습니다. 자세한 한계는 맨 아래에 정리했습니다.

# %%
# 1. 경로 설정 (src 폴더 접근용)
import sys, os
sys.path.append(os.path.abspath('..'))

# 2. Autoreload 설정 (src 내부의 py 파일 수정 시 즉시 반영)
# %load_ext autoreload
# %autoreload 2

# 경고 메시지 무시 (선택 사항)
import warnings
warnings.filterwarnings('ignore')

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats

from src.preprocessing.build_next_ab_dataset import (
    build_event_dataset_for_events,
    filter_to_same_batter_rematch,
    identify_extra_base_hit_events,
)
from src.preprocessing.research_sample import KEY_COLUMNS, list_research_csvs
from src.visualization.pitch_type_reuse_plots import LABELS, XBH_COLOR, _use_korean_font

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 60)
_use_korean_font()

# %% [markdown]
# ## 1. 팀 CSV 읽기와 재대결 이벤트
#
# `data/raw` 없이 팀 research CSV를 그대로 읽습니다. 기존 이벤트 빌더는 점수 컬럼을 요구하므로 0으로 채워 호출하고
# 결과의 `score_diff`는 버립니다(이 노트북에서는 점수를 쓰지 않습니다).

# %%
MEASURES = ['plate_x', 'plate_z', 'pfx_x', 'pfx_z', 'release_speed', 'release_spin_rate', 'release_extension']
LOAD_COLUMNS = [
    'game_pk', 'game_date', 'at_bat_number', 'pitch_number', 'pitcher', 'player_name', 'batter',
    'stand', 'p_throws', 'pitch_type', 'events', 'balls', 'strikes', 'outs_when_up',
    'inning', 'inning_topbot', *MEASURES,
]


def load_team_csv() -> pd.DataFrame:
    frames = [
        pd.read_csv(path, usecols=LOAD_COLUMNS, encoding='utf-8-sig', low_memory=False)
        for path in list_research_csvs()
    ]
    pitches = pd.concat(frames, ignore_index=True)
    pitches['season'] = pd.to_datetime(pitches['game_date']).dt.year
    pitches['home_score'] = 0  # CSV에 점수 없음: 빌더 호출용 자리 채움 (score_diff는 곧 버림)
    pitches['away_score'] = 0
    return pitches


pitches = load_team_csv()
assert len(pitches) == 3_592_302 and pitches['pitcher'].nunique() == 1_067
assert not pitches.duplicated(KEY_COLUMNS).any()

xbh_all = identify_extra_base_hit_events(pitches)
candidates = filter_to_same_batter_rematch(pitches, xbh_all)
events = build_event_dataset_for_events(pitches, candidates, next_pa_mode='same_batter').drop(columns='score_diff')
assert len(xbh_all) == 70_319 and len(events) == 27_391

reused = events[events['has_next_ab'] & (events['reused_same_type'] == 1)].reset_index(drop=True)
print(f'장타 {len(xbh_all):,}건 -> 같은 타자 재대결 {len(events):,}건 -> 같은 구종 재사용 {len(reused):,}건')
assert len(reused) == 12_459, len(reused)
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -4
```
Expected: `장타 70,319건 -> 같은 타자 재대결 27,391건 -> 같은 구종 재사용 12,459건`. 마지막 `assert`가 실패하면 **멈추고 실제 건수를 사용자에게 보고한다**(설계 문서에는 12,459건으로 적혀 있다).

- [ ] **Step 2: 투구 수집 테스트 셀 먼저 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/02_DY_rematch_pitch_execution_viz.py <<'EOF'

# %% test: collect_pa_pitches
_px = pd.DataFrame({
    'game_pk': [1] * 6, 'pitcher': [10] * 6,
    'pitch_type': ['SL', 'FF', 'SL', 'SL', 'FF', 'SL'],
    'at_bat_number': [5, 5, 5, 9, 9, 9], 'pitch_number': [1, 2, 3, 1, 2, 3],
    'plate_x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'plate_z': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
    'pfx_x': [0.0] * 6, 'pfx_z': [0.0] * 6, 'release_speed': [85.0, 95.0, 86.0, 87.0, 96.0, 88.0],
    'release_spin_rate': [2500.0] * 6, 'release_extension': [6.0] * 6,
})
_ev = pd.DataFrame({
    'game_pk': [1], 'pitcher': [10], 'hit_pitch_type': ['SL'], 'p_throws': ['R'], 'stand': ['L'],
    'at_bat_number': [5], 'event_pitch_number': [3], 'next_at_bat_number': [9.0],
})
_out = collect_pa_pitches(_px, _ev)
assert len(_out) == 4
assert sorted(_out['phase'].tolist()) == ['hit_pa', 'hit_pa', 'rematch_pa', 'rematch_pa']
assert _out.loc[_out['is_event_pitch'], 'plate_x'].tolist() == [0.3]      # 장타가 된 투구 (5타석 3구)
assert set(_out['release_speed']) == {85.0, 86.0, 87.0, 88.0}            # FF는 제외
assert _out['p_throws'].eq('R').all() and _out['event_id'].eq(0).all()
print('collect_pa_pitches 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: FAIL with `NameError: name 'collect_pa_pitches' is not defined`

- [ ] **Step 3: 구현 셀을 테스트 셀 바로 앞에 삽입**

Edit tool로 `$NB_SRC/02_DY_rematch_pitch_execution_viz.py`에서:

`old_string`:
```
# %% test: collect_pa_pitches
```
`new_string`:
```
# %% [markdown]
# ## 2. 같은 구종 투구 모으기
#
# 이벤트마다 (1) 장타 타석에서 그 투수가 던진 같은 구종 투구 전부(`hit_pa`, 실제 장타가 된 투구는 `is_event_pitch`)와
# (2) 재대결 타석에서 그 투수가 던진 같은 구종 투구 전부(`rematch_pa`)를 모읍니다.

# %% collect_pa_pitches
def collect_pa_pitches(pitches_: pd.DataFrame, events_: pd.DataFrame) -> pd.DataFrame:
    keys = ['game_pk', 'pitcher']
    ev = events_[keys + ['hit_pitch_type', 'p_throws', 'stand', 'at_bat_number', 'event_pitch_number', 'next_at_bat_number']].copy()
    ev['event_id'] = np.arange(len(ev))
    ev['next_at_bat_number'] = ev['next_at_bat_number'].astype(int)
    ev = ev.rename(columns={'at_bat_number': 'hit_at_bat'})
    px = pitches_[keys + ['pitch_type', 'at_bat_number', 'pitch_number', *MEASURES]].rename(
        columns={'pitch_type': 'hit_pitch_type'}
    )
    joined = ev.merge(px, on=keys + ['hit_pitch_type'], how='inner')  # 그 경기에서 그 투수의 같은 구종 투구 전부
    in_hit_pa = joined['at_bat_number'] == joined['hit_at_bat']
    in_rematch_pa = joined['at_bat_number'] == joined['next_at_bat_number']
    out = joined[in_hit_pa | in_rematch_pa].copy()
    out['phase'] = np.where(out['at_bat_number'] == out['hit_at_bat'], 'hit_pa', 'rematch_pa')
    out['is_event_pitch'] = (out['phase'] == 'hit_pa') & (out['pitch_number'] == out['event_pitch_number'])
    return out.drop(columns=['hit_at_bat', 'next_at_bat_number', 'event_pitch_number']).reset_index(drop=True)


# %% test: collect_pa_pitches
```

- [ ] **Step 4: 실제 데이터에 적용하는 셀을 추가하고 실행**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/02_DY_rematch_pitch_execution_viz.py <<'EOF'

# %%
pa = collect_pa_pitches(pitches, reused)
assert pa['event_id'].nunique() == len(reused)
assert pa.groupby('event_id')['is_event_pitch'].sum().eq(1).all(), '이벤트마다 장타가 된 투구가 정확히 1개여야 함'
assert (pa.groupby(['event_id', 'phase']).size().unstack('phase').notna().all().all()), '양쪽 타석 모두 투구가 있어야 함'
print(f'투구 {len(pa):,}개 (장타 타석 {(pa.phase == "hit_pa").sum():,}, 재대결 타석 {(pa.phase == "rematch_pa").sum():,})')
display(pa.groupby(['p_throws', 'hit_pitch_type'])['event_id'].nunique().unstack('p_throws').fillna(0).astype(int)
        .sort_values('R', ascending=False))
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -18
```
Expected: `collect_pa_pitches 테스트 통과` 이후 투구 개수 줄과, 손(L/R) × 구종별 이벤트 수 표가 나온다. 표는 앞서 확인한 값과 같아야 한다: 우투 FF 4,295, SI 1,546, SL 954, CH 567, FC 540, ST 359, CU 259, FS 170, KC 139 / 좌투 FF 1,816, SI 581, SL 295, CH 342, FC 263, ST 90, CU 139.

---

### Task 9: 노트북 2 — 이벤트별 변화와 수치표

**Files:**
- Modify: `$NB_SRC/02_DY_rematch_pitch_execution_viz.py`

- [ ] **Step 1: 변화·신뢰구간·요약표 테스트 셀 먼저 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/02_DY_rematch_pitch_execution_viz.py <<'EOF'

# %% test: deltas and summary
_d = event_deltas(_out)
assert np.isclose(_d.loc[0, 'plate_x'], 0.3) and np.isclose(_d.loc[0, 'release_speed'], 2.0)
assert _d.loc[0, 'hit_pitch_type'] == 'SL' and _d.loc[0, 'p_throws'] == 'R'

_m, _lo, _hi = mean_ci(pd.Series([1.0, 2.0, 3.0]))
assert np.isclose(_m, 2.0) and np.isclose(_hi - _m, 1.96 / np.sqrt(3)) and np.isclose(_m - _lo, 1.96 / np.sqrt(3))

_dd = pd.DataFrame({
    'p_throws': ['R', 'R'], 'hit_pitch_type': ['SL', 'SL'], 'stand': ['R', 'L'],
    **{c: [0.2, 0.4] for c in MEASURES},
})
_s = summary_table(_dd)
assert len(_s) == 1 and _s.loc[_s.index[0], 'n_events'] == 2
assert np.isclose(_s.loc[_s.index[0], '코스 x (ft) 평균변화'], 0.3)
assert np.isclose(_s.loc[_s.index[0], '무브먼트 x (in) 평균변화'], 0.3 * 12)   # ft -> in 환산
print('변화·요약표 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: FAIL with `NameError: name 'event_deltas' is not defined`

- [ ] **Step 2: 구현 셀을 테스트 셀 바로 앞에 삽입**

Edit tool로 `$NB_SRC/02_DY_rematch_pitch_execution_viz.py`에서:

`old_string`:
```
# %% test: deltas and summary
```
`new_string`:
```
# %% [markdown]
# ## 3. 이벤트별 변화와 수치표
#
# 이벤트마다 (재대결 타석 같은 구종 평균) − (장타 타석 같은 구종 평균)을 구하고, 구종 × 투수 손별로 평균 변화와 95% 신뢰구간(정규근사)을 냅니다.
# 코스는 ft, 무브먼트와 익스텐션은 인치(×12)로 환산합니다.

# %% deltas and summary
def event_deltas(pa_: pd.DataFrame) -> pd.DataFrame:
    means = pa_.groupby(['event_id', 'phase'])[MEASURES].mean().unstack('phase')
    delta = means.xs('rematch_pa', axis=1, level='phase') - means.xs('hit_pa', axis=1, level='phase')
    meta = pa_.groupby('event_id')[['p_throws', 'stand', 'hit_pitch_type']].first()
    return meta.join(delta)


def mean_ci(s: pd.Series) -> tuple[float, float, float]:
    s = s.dropna()
    n = len(s)
    m = s.mean()
    half = 1.96 * s.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    return m, m - half, m + half


DELTA_LABELS = {
    'plate_x': ('코스 x (ft)', 1), 'plate_z': ('코스 z (ft)', 1),
    'pfx_x': ('무브먼트 x (in)', 12), 'pfx_z': ('무브먼트 z (in)', 12),
    'release_speed': ('구속 (mph)', 1), 'release_spin_rate': ('회전수 (rpm)', 1),
    'release_extension': ('익스텐션 (in)', 12),
}


def summary_table(deltas_: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (hand, ptype), g in deltas_.groupby(['p_throws', 'hit_pitch_type']):
        row = {'p_throws': hand, 'hit_pitch_type': ptype, 'n_events': len(g)}
        for col, (label, scale) in DELTA_LABELS.items():
            m, lo, hi = mean_ci(g[col] * scale)
            row[f'{label} 평균변화'] = m
            row[f'{label} 95%CI'] = f'[{lo:.2f}, {hi:.2f}]'
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['p_throws', 'n_events'], ascending=[True, False]).reset_index(drop=True)


# %% test: deltas and summary
```

- [ ] **Step 3: 테스트 통과 확인**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: `변화·요약표 테스트 통과`

- [ ] **Step 4: 실제 수치표 셀 추가 및 저장**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/02_DY_rematch_pitch_execution_viz.py <<'EOF'

# %%
deltas = event_deltas(pa)
assert len(deltas) == len(reused)
table = summary_table(deltas)
display(table)
table.to_csv('../data/processed/rematch_execution_summary.csv', index=False)
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -30
```
Expected: 손 × 구종별 행(예: `L`, `R` 별 FF/SI/SL 등)과 각 지표의 평균 변화·신뢰구간이 출력된다. 수치가 극단적(예: 구속 변화가 ±5 mph 이상)이면 데이터 이상을 의심하고 보고한다.

---

### Task 10: 노트북 2 — 그림

**Files:**
- Modify: `$NB_SRC/02_DY_rematch_pitch_execution_viz.py`

- [ ] **Step 1: 그림 도우미 테스트 셀 먼저 추가**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/02_DY_rematch_pitch_execution_viz.py <<'EOF'

# %% test: plotting
assert density_level(np.array([[4.0, 3.0], [2.0, 1.0]]), 0.5) == 3.0


def _fake_pa(n_events: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        for phase, shift in (('hit_pa', 0.0), ('rematch_pa', 0.15)):
            for k in range(3):
                rows.append({
                    'event_id': e, 'phase': phase, 'p_throws': 'R', 'stand': 'R', 'hit_pitch_type': 'SL',
                    'pitch_number': k + 1, 'is_event_pitch': phase == 'hit_pa' and k == 2,
                    'plate_x': rng.normal(shift, 0.6), 'plate_z': rng.normal(2.2, 0.6),
                    'pfx_x': rng.normal(0.3, 0.1), 'pfx_z': rng.normal(0.1, 0.1),
                    'release_speed': rng.normal(85 + shift, 1.5), 'release_spin_rate': rng.normal(2400, 100),
                    'release_extension': rng.normal(6.3, 0.2),
                })
    return pd.DataFrame(rows)


_fig, _types = plot_hand_figure(_fake_pa(), 'R')
assert _types == ['SL']
with tempfile.TemporaryDirectory() as _tmp:
    _fig.savefig(Path(_tmp) / 'smoke.png', dpi=60)
    assert (Path(_tmp) / 'smoke.png').stat().st_size > 5_000
plt.close(_fig)
_fig2, _types2 = plot_hand_figure(_fake_pa(n_events=10), 'R', min_events=50)   # 표본이 모자라면 그릴 구종이 없다
assert _types2 == []
print('그림 도우미 테스트 통과')
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: FAIL with `NameError: name 'density_level' is not defined`

- [ ] **Step 2: 구현 셀을 테스트 셀 바로 앞에 삽입**

Edit tool로 `$NB_SRC/02_DY_rematch_pitch_execution_viz.py`에서:

`old_string`:
```
# %% test: plotting
```
`new_string`:
```
# %% [markdown]
# ## 4. 그림
#
# 투수 손별로 그림 한 장씩. 행 = 구종(손별로 이벤트 50건 이상인 구종만), 열 = 코스 · 무브먼트 · 구속.
#
# * 색: 장타 허용 타석 = 주황, 재대결 타석 = 파랑. 실제로 장타가 된 투구는 별표(★)
# * 점: 투구 하나하나(투명도 적용, 색마다 최대 1,500개 무작위 표시), 얇은 등고선: 밀집 영역 50%,
#   큰 원과 십자: 이벤트 단위 평균과 95% 신뢰구간
# * 스트라이크 존은 타자별 상하한(`sz_top/sz_bot`)이 데이터에 없어 **고정 사각형**(좌우 ±0.83 ft, 높이 1.5–3.5 ft)입니다. 실제 존은 타자마다 다릅니다.
# * 코스와 무브먼트는 포수 시점(오른쪽 = 1루 쪽)입니다. 타자 좌/우는 나누지 않고 섞어서 그립니다.

# %% plotting
ZONE_HALF_WIDTH, ZONE_Z = 0.83, (1.5, 3.5)
PHASE_COLORS = {'hit_pa': XBH_COLOR, 'rematch_pa': '#0072B2'}
PHASE_LABELS = {'hit_pa': '장타 허용 타석', 'rematch_pa': '재대결 타석'}
PITCH_LABELS = {
    **LABELS, 'FS': '스플리터 (FS)', 'KC': '너클커브 (KC)', 'SV': '슬러브 (SV)', 'KN': '너클볼 (KN)', 'FO': '포크볼 (FO)',
}
HAND_LABELS = {'R': '우투수', 'L': '좌투수'}
MAX_POINTS = 1500


def density_level(zz: np.ndarray, mass: float = 0.5) -> float:
    """밀도 격자 zz에서 전체 밀도의 `mass` 비율을 감싸는 등고선 수준."""
    order = np.sort(zz.ravel())[::-1]
    cumulative = np.cumsum(order) / order.sum()
    return float(order[np.searchsorted(cumulative, mass)])


def draw_density_contour(ax, x, y, color, mass: float = 0.5, grid: int = 70) -> None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 30 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return
    kde = stats.gaussian_kde(np.vstack([x, y]))
    pad_x, pad_y = 0.15 * np.ptp(x), 0.15 * np.ptp(y)
    gx, gy = np.meshgrid(
        np.linspace(x.min() - pad_x, x.max() + pad_x, grid), np.linspace(y.min() - pad_y, y.max() + pad_y, grid)
    )
    zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contour(gx, gy, zz, levels=[density_level(zz, mass)], colors=[color], linewidths=1.6)


def _panel_xy(ax, g: pd.DataFrame, xcol: str, ycol: str, scale: float, seed: int, zone: bool) -> None:
    for phase in ('hit_pa', 'rematch_pa'):
        p = g[g['phase'] == phase].dropna(subset=[xcol, ycol])
        if p.empty:
            continue
        color = PHASE_COLORS[phase]
        shown = p.sample(min(len(p), MAX_POINTS), random_state=seed)
        ax.scatter(shown[xcol] * scale, shown[ycol] * scale, s=8, alpha=0.18, color=color, linewidths=0)
        draw_density_contour(ax, shown[xcol] * scale, shown[ycol] * scale, color)
        stars = p[p['is_event_pitch']]
        if len(stars):
            stars = stars.sample(min(len(stars), MAX_POINTS), random_state=seed)
            ax.scatter(
                stars[xcol] * scale, stars[ycol] * scale, s=34, marker='*', color=color,
                edgecolors='black', linewidths=0.4, alpha=0.9, zorder=3,
            )
        per_event = p.groupby('event_id')[[xcol, ycol]].mean() * scale
        mx, lx, hx = mean_ci(per_event[xcol])
        my, ly, hy = mean_ci(per_event[ycol])
        ax.errorbar(
            mx, my, xerr=[[mx - lx], [hx - mx]], yerr=[[my - ly], [hy - my]], fmt='o', ms=8, color=color,
            mec='black', mew=1.2, capsize=3, lw=2, zorder=4,
        )
    if zone:
        ax.add_patch(Rectangle(
            (-ZONE_HALF_WIDTH, ZONE_Z[0]), 2 * ZONE_HALF_WIDTH, ZONE_Z[1] - ZONE_Z[0],
            fill=False, ec='black', lw=1.6, zorder=2,
        ))
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0.0, 5.0)
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.25)


def _panel_velocity(ax, g: pd.DataFrame) -> None:
    low, high = g['release_speed'].quantile([0.005, 0.995])
    grid = np.linspace(low - 1, high + 1, 200)
    per_event = {}
    for phase in ('hit_pa', 'rematch_pa'):
        v = g.loc[g['phase'] == phase, 'release_speed'].dropna()
        if len(v) < 5:
            continue
        color = PHASE_COLORS[phase]
        density = stats.gaussian_kde(v)(grid)
        ax.fill_between(grid, density, color=color, alpha=0.25)
        ax.plot(grid, density, color=color, lw=1.8)
        ax.axvline(v.mean(), color=color, lw=1.6, ls='--')
        per_event[phase] = g[g['phase'] == phase].groupby('event_id')['release_speed'].mean()
    if len(per_event) == 2:
        m, lo, hi = mean_ci((per_event['rematch_pa'] - per_event['hit_pa']).dropna())
        ax.text(0.02, 0.95, f'평균 변화 {m:+.2f} mph\n95% CI [{lo:+.2f}, {hi:+.2f}]',
                transform=ax.transAxes, va='top', fontsize=9)
    ax.set_yticks([])
    ax.grid(alpha=0.25)


def plot_hand_figure(pa_: pd.DataFrame, hand: str, min_events: int = 50, seed: int = 0):
    """투수 손 하나의 그림 (행 = 구종, 열 = 코스·무브먼트·구속). (figure, 그린 구종 목록) 반환."""
    sub = pa_[pa_['p_throws'] == hand]
    counts = sub.groupby('hit_pitch_type')['event_id'].nunique().sort_values(ascending=False)
    types = counts[counts >= min_events].index.tolist()
    fig, axes = plt.subplots(max(len(types), 1), 3, figsize=(13.5, 3.7 * max(len(types), 1)), squeeze=False)
    for row, ptype in enumerate(types):
        g = sub[sub['hit_pitch_type'] == ptype]
        _panel_xy(axes[row, 0], g, 'plate_x', 'plate_z', 1.0, seed, zone=True)
        _panel_xy(axes[row, 1], g, 'pfx_x', 'pfx_z', 12.0, seed, zone=False)
        _panel_velocity(axes[row, 2], g)
        axes[row, 0].set_ylabel(f'{PITCH_LABELS.get(ptype, ptype)}\n(장타 {counts[ptype]:,}건)\nplate_z (ft)', fontsize=10)
    if types:
        axes[0, 0].set_title('코스 (포수 시점, ft)')
        axes[0, 1].set_title('무브먼트 (포수 시점, 인치)')
        axes[0, 2].set_title('구속 (mph)')
        axes[-1, 0].set_xlabel('plate_x (ft, 오른쪽 = 1루 쪽)')
        axes[-1, 1].set_xlabel('pfx_x (인치)')
        axes[-1, 2].set_xlabel('release_speed (mph)')
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PHASE_COLORS[p], markeredgecolor='black', ms=10, label=PHASE_LABELS[p])
        for p in PHASE_COLORS
    ]
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='#999999', markeredgecolor='black', ms=14, label='실제 장타가 된 투구'))
    fig.suptitle(f'{HAND_LABELS[hand]}: 장타 허용 타석 vs 같은 타자 재대결 타석 (같은 구종)', fontsize=14, y=0.995)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.972), ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, types


# %% test: plotting
```

- [ ] **Step 3: 테스트 통과 확인**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -3
```
Expected: `그림 도우미 테스트 통과`

- [ ] **Step 4: 실제 그림 생성 셀 추가 (우투·좌투, 파일 저장)**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cat >> $NB_SRC/02_DY_rematch_pitch_execution_viz.py <<'EOF'

# %%
FIGURE_DIR = Path('../data/processed/figures')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

for hand, name in (('R', 'RHP'), ('L', 'LHP')):
    fig, types = plot_hand_figure(pa, hand)
    fig.savefig(FIGURE_DIR / f'rematch_execution_{name}.png', dpi=130)
    print(f'{HAND_LABELS[hand]}: 그린 구종 {types}')
    plt.show()

drawn = {h: set(pa[pa['p_throws'] == h].groupby('hit_pitch_type')['event_id'].nunique().loc[lambda s: s >= 50].index) for h in 'RL'}
omitted = (
    pa.groupby(['p_throws', 'hit_pitch_type'])['event_id'].nunique().rename('n_events').reset_index()
    .loc[lambda d: d.apply(lambda r: r['hit_pitch_type'] not in drawn[r['p_throws']], axis=1)]
)
print('이벤트 50건 미만이라 그리지 않은 구종 (수치표에는 있음):')
display(omitted)
EOF
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && time MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py 2>&1 | grep -v -E 'API mode|ABI mode' | tail -12; ls -l ../data/processed/figures | grep rematch
```
Expected: `우투수: 그린 구종 ['FF', 'SI', 'SL', 'CH', 'FC', 'ST', 'CU', 'FS', 'KC']`, `좌투수: 그린 구종 ['FF', 'SI', 'CH', 'SL', 'FC', 'CU', 'ST']`(순서는 이벤트 수 내림차순), 그리지 않은 구종 표, 그리고 `rematch_execution_RHP.png`, `rematch_execution_LHP.png` 두 파일이 보인다.

- [ ] **Step 5: 그림을 눈으로 확인하고 다듬기**

Read tool로 `data/processed/figures/rematch_execution_RHP.png` 와 `rematch_execution_LHP.png` 를 열어 확인한다. 점검 항목: 범례가 제목과 겹치지 않는가, 스트라이크 존 사각형이 보이는가, 주황(장타 타석)과 파랑(재대결 타석) 구분이 되는가, 별표가 보이는가, 행 라벨(구종과 건수)이 잘리지 않는가, 구속 패널의 평균 변화 글씨가 곡선을 가리지 않는가. 문제가 있으면 `plot_hand_figure`의 `bbox_to_anchor`, `rect`, `figsize`, 글꼴 크기를 조정하고 다시 실행한다. 데이터로 결론을 바꾸는 조정은 하지 않는다.

---

### Task 11: 노트북 2 — 해석과 한계

**Files:**
- Modify: `$NB_SRC/02_DY_rematch_pitch_execution_viz.py`

- [ ] **Step 1: 결과를 읽고 해석·한계 셀 추가**

Task 9의 수치표와 Task 10의 그림을 읽고, 파일 끝에 마크다운 셀을 추가한다. **숫자는 실제 출력값을 옮긴다.** 구성:

1. 관찰 요약: 구종·손별로 가장 뚜렷한 변화(코스 x·z, 무브먼트, 구속)를 신뢰구간이 0을 포함하는지 여부와 함께 2–4줄로 적는다. 신뢰구간이 0을 포함하면 "뚜렷한 변화 없음"으로 쓴다.
2. 한계(설계 문서 5.4절)를 그대로 포함한다.
3. 고정 존, 타자 좌/우 혼합도 한계에 넣는다.

한계 부분의 고정 문안(그대로 사용):
```python
# %% [markdown]
# ## 6. 한계
#
# * **대조군이 없습니다.** 변화가 "장타에 대한 반응"이라고 단정할 수 없습니다. 예를 들어 구속은 경기가 진행되며 대조군도 평균 0.33 mph 떨어집니다(README).
# * **선택 편향:** 장타가 된 투구는 실투(가운데로 몰린 공 등)였을 가능성이 있어, 그 투구를 포함한 장타 타석 평균은 평소와 다를 수 있습니다(별표로 구분).
# * **고정 스트라이크 존:** 타자별 존 상하한이 데이터에 없어 고정 사각형으로 그렸습니다.
# * **타자 좌/우 혼합:** 좌·우타자를 섞어 그려서 코스 분포가 넓어 보일 수 있습니다. 한 이벤트의 두 타석은 같은 타자라 이벤트 안 비교는 공정합니다.
```
관찰 요약 셀은 다음 형식으로 앞에 붙인다.
```python
# %% [markdown]
# ## 5. 관찰 (수치표 기준)
#
# * (실제 수치로 채운 문장들 — 구종·손, 평균 변화, 95% CI)
```

- [ ] **Step 2: 마지막 실행 확인**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2/notebooks && MPLBACKEND=Agg ../.venv/bin/python $NB_SRC/02_DY_rematch_pitch_execution_viz.py > /dev/null 2>&1; echo "exit code: $?"
```
Expected: `exit code: 0`

---

### Task 12: 최종 검증과 인계

**Files:**
- Create: `notebooks/01_DY_situation_matched_control.ipynb`, `notebooks/02_DY_rematch_pitch_execution_viz.ipynb`

- [ ] **Step 1: 원본 → ipynb 변환**

Run:
```bash
NB_SRC=/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 \
&& .venv/bin/python $NB_SRC/to_ipynb.py $NB_SRC/01_DY_situation_matched_control.py notebooks/01_DY_situation_matched_control.ipynb \
&& .venv/bin/python $NB_SRC/to_ipynb.py $NB_SRC/02_DY_rematch_pitch_execution_viz.py notebooks/02_DY_rematch_pitch_execution_viz.ipynb
```
Expected: `wrote notebooks/01_DY_...ipynb`, `wrote notebooks/02_DY_...ipynb`

- [ ] **Step 2: headless로 처음부터 끝까지 실행 (작업 트리에 결과를 남김)**

Run (각 노트북은 수 분–10분 이상 걸릴 수 있으니 백그라운드 실행을 권장):
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 \
&& .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/01_DY_situation_matched_control.ipynb 2>&1 | tail -2 \
&& .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/02_DY_rematch_pitch_execution_viz.ipynb 2>&1 | tail -2
```
Expected: 각각 `Writing ... bytes to notebooks/...ipynb`, 에러 없음. 실패하면 오류 셀을 원본 `.py`에서 고쳐 Step 1부터 다시 한다.

- [ ] **Step 3: 저장소 상태 확인**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 && git status --short && .venv/bin/python -m pytest -q 2>&1 | tail -2 && du -h notebooks/*.ipynb
```
Expected: 변경은 ` M requirements.txt`, `?? notebooks/01_DY_...ipynb`, `?? notebooks/02_DY_...ipynb`, `?? docs/superpowers/plans/...`, `?? docs/superpowers/specs/` 뿐이고(`.venv`, `data/` 없음), `86 passed`. 노트북 크기는 실행 결과 때문에 클 수 있다(그림 포함, 수 MB).

- [ ] **Step 4: 사용자에게 보고**

다음을 정리해 보고한다: (1) 환경 검증 결과(README 수치 재현 여부), (2) 실험 1 사다리 요약표와 해석, (3) 실험 2 그림 2장 위치와 관찰, (4) 노트북에 실행 결과가 남아 있고 커밋 시점에 출력을 지운다는 점, (5) 커밋 여부를 묻는다. **사용자가 커밋을 요청하기 전에는 커밋하지 않는다.**

- [ ] **Step 5: (사용자가 요청한 경우에만) 출력 제거 후 커밋**

Run:
```bash
cd /Users/dongyunkwak/GitHub/9th-first-project-baseball-2 \
&& .venv/bin/jupyter nbconvert --clear-output --inplace notebooks/01_DY_situation_matched_control.ipynb notebooks/02_DY_rematch_pitch_execution_viz.ipynb \
&& git add requirements.txt notebooks/01_DY_situation_matched_control.ipynb notebooks/02_DY_rematch_pitch_execution_viz.ipynb \
&& git commit -m "feat: add situation-matched control and rematch pitch-execution notebooks" \
       -m "Notebook 01 rebuilds the placebo group matched on base-out state, inning and count from the team CSV runner columns and compares the pure reduction and mixed-model odds ratio across a matching ladder. Notebook 02 plots same-pitch location, movement and velocity in the hit plate appearance versus the rematch plate appearance by pitcher hand. Notebook outputs are cleared per notebooks/README.md." \
&& git log -1 --format=%B | grep -i -c -E 'claude|anthropic|co-authored'
```
Expected: 커밋이 만들어지고 마지막 줄 `0`(Claude 관련 문구 없음). 설계·계획 문서(`docs/superpowers/...`)를 함께 커밋할지는 사용자에게 따로 묻는다. push는 요청이 있을 때만 한다.

---

## 자체 검토 (계획 작성 후)

**설계 문서 대비 커버리지**

| 설계 문서 항목 | 계획의 위치 |
|---|---|
| 2.1 CSV만 사용, 점수 없음 처리 | Task 3 Step 1, Task 8 Step 1 (0 채움 후 `score_diff` 버림) |
| 2.2 환경 세팅 | Task 1 |
| 2.4 환경 검증(70,319 / 27,391 / 27,371 / M0 수치 / pytest) | Task 4 Step 1, Task 6 Step 1, Task 1 Step 7 |
| 3 노트북 규칙(파일명, 첫 셀, 상대 경로, 출력 지우기, headless 실행) | Task 3 Step 1, Task 12 |
| 4.1 상황 변수 | Task 3 Steps 2–4 (`add_situation_columns`), Task 4 Steps 2–4 (`attach_*`) |
| 4.3 사다리 M0/M1/M2/C0, 90% 규칙 | Task 5 Steps 4–5, Task 7 Step 4 |
| 4.4 보고 항목 1–5 | 유지율(Task 5 Step 4, Task 7 Step 4), SMD(Task 6 Step 2), 순수 감소·오즈비(Task 7 Step 4), 층화(Task 7 Step 5) |
| 4.5 해석 원칙 | Task 7 Step 6 |
| 5.1–5.4 실험 2 | Tasks 8–11 |
| 6 완료 기준 | Task 12 |

**플레이스홀더 점검:** 해석 셀(Task 7 Step 6, Task 11 Step 1)만 실제 결과 수치로 채워야 하며, 형식과 판단 규칙을 그 자리에 적어 두었다. 나머지 단계에는 완전한 코드와 명령이 있다.

**이름 일관성 점검:** `SITUATION_COLUMNS`, `NEXT_COLUMNS`, `add_situation_columns`, `attach_event_situation`, `attach_next_situation`, `smd`, `balance_table`, `usable`, `pure_reduction`, `runner_group`, `build_step`, `STEPS`, `pairs`, `fit_mixed`, `key_terms`, `BASE_FORMULA`, `C0_FORMULA`, `stratified_reduction`(노트북 1) / `collect_pa_pitches`, `event_deltas`, `mean_ci`, `summary_table`, `density_level`, `draw_density_contour`, `plot_hand_figure`, `MEASURES`(노트북 2)는 정의한 태스크와 사용하는 태스크에서 같은 이름·시그니처를 씁니다.
