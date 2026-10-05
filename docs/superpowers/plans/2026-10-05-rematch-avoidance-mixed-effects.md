# Rematch Avoidance Mixed-Effects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/07_DY_rematch_avoidance_mixed_effects.ipynb`, which tests whether avoiding the pitch type that allowed an extra-base hit, in the same-batter rematch, predicts a better outcome — at the event level (27,391 rematch events, no per-pitcher aggregation), with pitcher as a random intercept.

**Architecture:** Reuse notebook 3's CSV-loading, situation-feature, rematch-eventset, and GBM/RVAE code verbatim (already tested there). Reuse this project's existing R/lme4 bridge (`src/analysis/glmer_runner.py`, `src/analysis/mixed_effects_model.py`) for both a gaussian model (D: `rvae ~ avoided + ...`, via the already-existing `fit_lmer`/`extract_lmer_fixed_effects`) and a binomial model (E: `allowed_xbh_again ~ avoided + ...`, via the already-existing `fit_glmer`/`extract_fixed_effects_table`). The only new `src/` code is a small variance-component extractor and a gaussian-ICC formula; everything else new lives in the notebook source.

**Tech Stack:** pandas, scikit-learn (`HistGradientBoostingRegressor`, reused from notebook 3), R `lme4` via rpy2 (`fit_glmer`, `fit_lmer` — both already implemented).

**Spec:** `docs/superpowers/specs/2026-10-05-rematch-avoidance-mixed-effects-design.md`

## Global Constraints

- No per-pitcher minimum rematch-event threshold — every `has_next_ab=True` rematch event is included (spec §3); do not reintroduce the ≥10-events filter used in notebooks 3/4/6.
- Model formulas, exact (spec §4):
  - `(D)` `rvae ~ avoided + balls + strikes + outs_when_up + score_diff + platoon_match + pitch_family + factor(season) + (1 | pitcher)`, fit via `lmer` (gaussian).
  - `(E)` `allowed_xbh_again ~ avoided + balls + strikes + outs_when_up + score_diff + platoon_match + pitch_family + factor(season) + (1 | pitcher)`, fit via `glmer` (binomial).
- `pitcher_id` is never a fixed effect (only the `(1 | pitcher)` grouping factor); no execution-quality features (release_speed/pfx_x/pfx_z/plate_x/plate_z); no general pitch-recommendation framing.
- RVAE (`(D)`'s target) comes from notebook 3's GBM (situation + `pitch_type` + `season_usage_rate` only) — do not retrain with a different feature set.
- Reuse existing R infra by name — `fit_glmer`, `fit_lmer`, `extract_fixed_effects_table`, `extract_lmer_fixed_effects`, `check_convergence` (all in `src/analysis/glmer_runner.py`) — do not reimplement them.
- Do not modify notebooks 3–6 or their scratchpad sources.
- Notebook source lives at `$NB_SRC/07_DY_rematch_avoidance_mixed_effects.py` (percent-format), where `$NB_SRC` = `/private/tmp/claude-501/-Users-dongyunkwak-GitHub-9th-first-project-baseball-2/bc3d8435-d9b9-44b2-a32a-5ea9d582679f/scratchpad/nb_src`; convert via `$NB_SRC/to_ipynb.py` and execute headlessly via `jupyter nbconvert --to notebook --execute --inplace` before considering any task that touches it complete, same as notebooks 3–6.
- No Claude/Anthropic attribution anywhere (commit messages, notebook content).

## Review Focus

- **Convergence/singular fit isn't silently swallowed.** `check_convergence()`'s output (including `isSingular=TRUE`) must be printed and reported for both models regardless of what it says — a singular fit is an acceptable, reportable result per spec §4, not a task failure to hide.
- **Missing values are dropped explicitly, with a reported count, before fitting** — R's `na.omit` default would do this silently otherwise, exactly the failure mode `build_combined_model_dataset` already guards against elsewhere in this codebase.
- **`pitcher` must be a string/factor, not numeric**, or `(1 | pitcher)` fits a nonsense numeric-slope-like grouping instead of a categorical random intercept.
- **`pitch_family` must not leak a `None`/NaN category into `factor()`** — `map_pitch_family` returns `None` for an unmapped or missing `pitch_type`; such rows must be excluded (dropna), not pass through as a literal "None" factor level.
- **`allowed_xbh_again` must not silently read as 0 for rows with no known outcome.** A row missing `events` (or `woba_value`) must be excluded before labeling, not default to "no XBH" via `NaN.isin(...)` returning `False`.

---

### Task 1: Variance-component extraction for gaussian ICC

**Files:**
- Modify: `src/analysis/glmer_runner.py`
- Modify: `src/analysis/mixed_effects_model.py`
- Test: `tests/analysis/test_mixed_effects_model.py`

**Interfaces:**
- Consumes: nothing new (uses the R global `model` that `fit_glmer`/`fit_lmer` already set, same convention as every other `extract_*` function in `glmer_runner.py`).
- Produces: `extract_variance_components(grouping: str = "pitcher") -> dict[str, float | None]` in `glmer_runner.py`, returning `{f"{grouping}_intercept": <variance or None>, "residual": <variance or None>}` — `residual` is `None` when the current model has no `Residual` row in `VarCorr(model)` (true for a binomial `glmer` fit; present for a gaussian `lmer` fit). `compute_icc_gaussian(intercept_var: float, residual_var: float) -> float` in `mixed_effects_model.py`, returning `intercept_var / (intercept_var + residual_var)` — used later by Task 5; `compute_icc` (existing, logistic-specific) stays as-is for Task 6.

- [ ] **Step 1: Write the failing test for `compute_icc_gaussian`**

```python
def test_compute_icc_gaussian_matches_known_ratio():
    assert math.isclose(compute_icc_gaussian(intercept_var=2.0, residual_var=6.0), 0.25)

def test_compute_icc_gaussian_zero_intercept_is_zero():
    assert compute_icc_gaussian(intercept_var=0.0, residual_var=4.0) == 0.0
```

Add `import math` at the top of `tests/analysis/test_mixed_effects_model.py` if not already present, and add `compute_icc_gaussian` to the existing import from `src.analysis.mixed_effects_model`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analysis/test_mixed_effects_model.py -k icc_gaussian -v`
Expected: FAIL with `ImportError` / `cannot import name 'compute_icc_gaussian'`

- [ ] **Step 3: Implement `compute_icc_gaussian` in `src/analysis/mixed_effects_model.py`**

Place it next to the existing `compute_icc`, with a one-line docstring noting it's the gaussian-response analogue (ICC = intercept variance / (intercept variance + residual variance), vs. `compute_icc`'s logistic-link π²/3 formula).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/analysis/test_mixed_effects_model.py -k icc_gaussian -v`
Expected: PASS (2/2)

- [ ] **Step 5: Implement `extract_variance_components` in `src/analysis/glmer_runner.py`**

Not unit-tested — this file's existing docstring policy (R/rpy2 integration-level, exercised by actually running against real data, not every contributor has R installed) applies here too; follow the same pattern as the neighboring `extract_pitcher_variance_components`/`extract_ranef_table` (read `VarCorr(model)` as an R data frame, convert via `R_CONVERTER`, filter rows by `grp`/`var1`/`var2`). Verify it by hand: in a scratch Python shell, `fit_lmer` a tiny synthetic 2-column R model with a known random-intercept and residual variance (e.g. `y ~ 1 + (1|g)` on a small synthetic `pd.DataFrame`) and confirm the returned dict's values are in the right ballpark (R prints `VarCorr(model)` — compare by eye). Record the command and its output in the task's completion note; no automated test is expected for this step.

- [ ] **Step 6: Commit**

```bash
git add src/analysis/glmer_runner.py src/analysis/mixed_effects_model.py tests/analysis/test_mixed_effects_model.py
git commit -m "feat: add gaussian-model variance-component extraction for ICC"
```

---

### Task 2: Notebook scaffold — data load through GBM/RVAE

**Files:**
- Create: `$NB_SRC/07_DY_rematch_avoidance_mixed_effects.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (module-level names later tasks consume): `pitches` (all pitches, situation columns added, `season` present), `xbh` (rematch eventset, already filtered to `has_next_ab=True`, has `reused_same_type`, `pitcher`, `game_pk`, `next_at_bat_number`, `events` [the *original hit's* event, not used downstream], `season`), `season_usage`, `decisive_table` (all-decisive-pitch table with `season_usage_rate` attached), `final_model` (the fitted `HistGradientBoostingRegressor`), `MODEL_FEATURES`, `CATEGORICAL_FEATURES`, `compute_rvae(df, model) -> pd.Series`.

- [ ] **Step 1: Copy notebook 3's Sections 1–3 and 6–7 into the new file**

Read `$NB_SRC/03_DY_run_value_gbm_pitcher_tendency.py` and copy, verbatim, the markdown + code cells for: CSV reading (`LOAD_COLUMNS`, `load_team_csv`), `add_situation_columns` (+ its test), rematch eventset construction (`xbh = build_xbh_rematch(pitches, season_usage); xbh = xbh[xbh['has_next_ab']].copy()` — skip the avoidance-rate/tercile-grouping cells, not needed here), the GBM training table (`MODEL_FEATURES`, `CATEGORICAL_FEATURES`, `build_decisive_pitch_table` + its test), and model fit/validation (`fit_run_value_model`, the train/validate cell, `final_model = fit_run_value_model(decisive_table, decisive_table['season'].unique())`), and `compute_rvae` (+ its test). Renumber markdown sections 1–4 to match this notebook's own structure (CSV / situation / 재대결 이벤트 / GBM 학습·검증); rewrite the top banner markdown per the spec (title, purpose, link to this spec file, not notebook 3's).

- [ ] **Step 2: Run the script to verify this section executes cleanly**

Run: `MPLBACKEND=Agg ../.venv/bin/python -u $NB_SRC/07_DY_rematch_avoidance_mixed_effects.py` (from `notebooks/`)
Expected: all copied small-fixture tests print "통과"; script ends (nothing after this section yet) without a traceback; real-data prints match notebook 3's own (3,592,302 pitches; 27,391 rematch events; GBM MAE ≈0.3989 / R²≈0.057).

- [ ] **Step 3: Commit**

```bash
git add -A  # (no repo files changed yet besides the scratchpad source, which is git-ignored by convention — commit anything in-repo this step touched, if any; otherwise skip the commit)
```

If nothing under the repo root changed (scratchpad is outside the repo), skip this step's commit — note that in the task-done ledger instead.

---

### Task 3: Rematch-outcome table and modeling-dataset builder

**Files:**
- Modify: `$NB_SRC/07_DY_rematch_avoidance_mixed_effects.py`

**Interfaces:**
- Consumes: `xbh`, `decisive_table`, `MODEL_FEATURES` (Task 2).
- Produces: `find_rematch_outcome_pitch(xbh_: pd.DataFrame, pitches_sit: pd.DataFrame) -> pd.DataFrame` (one row per `xbh_` row, columns `['game_pk', 'pitcher', 'season', 'at_bat_number', *MODEL_FEATURES, 'platoon_match', 'events', 'woba_value', 'reused_same_type']`); `build_avoidance_outcome_dataset(found: pd.DataFrame) -> pd.DataFrame` (adds `avoided`, `pitch_family`, `allowed_xbh_again`, casts `pitcher` to `str`, drops rows missing any of `rvae`-independent required columns, returns the trimmed frame — `rvae` itself is added in Task 4 once the model is available).

- [ ] **Step 1: Write the failing test for `find_rematch_outcome_pitch`**

```python
def test_find_rematch_outcome_pitch_carries_platoon_events_and_reused_flag():
    _xbh_fix = pd.DataFrame({
        'game_pk': [100], 'pitcher': [10], 'next_at_bat_number': [5.0], 'reused_same_type': [1.0],
    })
    _pt_fix = pd.DataFrame({
        'game_pk': [100], 'pitcher': [10], 'season': [2021], 'at_bat_number': [5],
        'woba_value': [0.9], 'events': ['double'], 'pitch_type': ['SL'], 'stand': ['R'], 'p_throws': ['R'],
        'balls': [1], 'strikes': [2], 'outs_when_up': [0], 'runners_n': [0], 'risp': [0], 'score_diff': [0],
        'season_usage_rate': [0.5], 'platoon_match': [1.0],
    })
    found = find_rematch_outcome_pitch(_xbh_fix, _pt_fix)
    assert len(found) == 1
    assert found.iloc[0]['events'] == 'double' and found.iloc[0]['platoon_match'] == 1.0
    assert found.iloc[0]['reused_same_type'] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg ../.venv/bin/python -u $NB_SRC/07_DY_rematch_avoidance_mixed_effects.py` (temporarily add the test cell at the end; or test interactively) — confirm it fails with `NameError: name 'find_rematch_outcome_pitch' is not defined`.
Expected: FAIL, function not defined.

- [ ] **Step 3: Implement `find_rematch_outcome_pitch`**

Same shape as notebook 3's `find_rematch_decisive_pitch`, extended: the `decisive` selection list gains `'platoon_match'` and `'events'`; the `keys` frame gains `'reused_same_type'` (carried from `xbh_`) alongside `game_pk`/`pitcher`/`next_at_bat_number`→`at_bat_number`. Still a `how='left'` merge (keep NaN rows for unmatched rematches, same as notebook 3, so the missing-pitch diagnostic in Task 4 has something to count).

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS.

- [ ] **Step 5: Write the failing tests for `build_avoidance_outcome_dataset`**

```python
def test_build_avoidance_outcome_dataset_derives_avoided_and_xbh_again():
    found = pd.DataFrame({
        'pitcher': [10, 20], 'season': [2021, 2021], 'reused_same_type': [1.0, 0.0],
        'pitch_type': ['SL', 'FF'], 'events': ['double', 'field_out'], 'woba_value': [0.9, 0.0],
        'balls': [1, 0], 'strikes': [2, 1], 'outs_when_up': [0, 1], 'score_diff': [0, -1],
        'platoon_match': [1.0, 0.0], 'season_usage_rate': [0.5, 0.4], 'stand': ['R', 'L'], 'p_throws': ['R', 'R'],
        'runners_n': [0, 1], 'risp': [0, 0],
    })
    out = build_avoidance_outcome_dataset(found)
    assert out['avoided'].tolist() == [0, 1]  # reused -> not avoided; didn't reuse -> avoided
    assert out['allowed_xbh_again'].tolist() == [1, 0]
    assert out['pitch_family'].tolist() == ['breaking', 'fastball']
    assert out['pitcher'].dtype == object and out['pitcher'].tolist() == ['10', '20']

def test_build_avoidance_outcome_dataset_drops_missing_outcome_not_zero_fills():
    found = pd.DataFrame({
        'pitcher': [10], 'season': [2021], 'reused_same_type': [1.0], 'pitch_type': ['SL'],
        'events': [None], 'woba_value': [np.nan], 'balls': [1], 'strikes': [2], 'outs_when_up': [0],
        'score_diff': [0], 'platoon_match': [1.0], 'season_usage_rate': [0.5], 'stand': ['R'], 'p_throws': ['R'],
        'runners_n': [0], 'risp': [0],
    })
    out = build_avoidance_outcome_dataset(found)
    assert len(out) == 0  # missing events/woba_value row must be dropped, not default allowed_xbh_again=0

def test_build_avoidance_outcome_dataset_drops_unmapped_pitch_type_not_none_category():
    found = pd.DataFrame({
        'pitcher': [10, 20], 'season': [2021, 2021], 'reused_same_type': [1.0, 0.0],
        'pitch_type': [None, 'FF'], 'events': ['field_out', 'field_out'], 'woba_value': [0.0, 0.0],
        'balls': [1, 0], 'strikes': [2, 1], 'outs_when_up': [0, 1], 'score_diff': [0, -1],
        'platoon_match': [1.0, 0.0], 'season_usage_rate': [0.5, 0.4], 'stand': ['R', 'L'], 'p_throws': ['R', 'R'],
        'runners_n': [0, 1], 'risp': [0, 0],
    })
    out = build_avoidance_outcome_dataset(found)
    assert len(out) == 1 and out.iloc[0]['pitcher'] == '20'  # pitch_type=None -> pitch_family=None -> dropped
    assert out['pitch_family'].tolist() == ['fastball']
```

- [ ] **Step 6: Run tests to verify they fail**

Expected: FAIL, `NameError`.

- [ ] **Step 7: Implement `build_avoidance_outcome_dataset`**

`avoided = 1 - reused_same_type` (nullable-safe); `pitch_family = found['pitch_type'].map(map_pitch_family)` (import `map_pitch_family` from `src.preprocessing.pitch_family`); `allowed_xbh_again = found['events'].isin(EXTRA_BASE_HIT_EVENTS).astype(int)` (import `EXTRA_BASE_HIT_EVENTS` from `src.preprocessing.build_next_ab_dataset`) computed *after* dropping rows with missing `events`/`woba_value`/`pitch_family`/`platoon_match`/any `MODEL_FEATURES` column (explicit `dropna(subset=[...])`, matching the Review Focus item); `pitcher = pitcher.astype(str)`.

- [ ] **Step 8: Run tests to verify they pass**

Expected: PASS (all three tests).

- [ ] **Step 9: Commit**

```bash
# scratchpad source is git-ignored; note completion in the task-done ledger instead of committing
```

---

### Task 4: Apply to real data

**Files:**
- Modify: `$NB_SRC/07_DY_rematch_avoidance_mixed_effects.py`

**Interfaces:**
- Consumes: `find_rematch_outcome_pitch`, `build_avoidance_outcome_dataset` (Task 3), `xbh`, `decisive_table`, `final_model`, `compute_rvae` (Task 2).
- Produces: `model_data` (the full modeling-ready `pd.DataFrame` with `rvae` and `allowed_xbh_again` both present, one row per usable rematch event), printed diagnostics.

- [ ] **Step 1: Write the real-data application cell**

```python
found = find_rematch_outcome_pitch(xbh, decisive_table)
missing = found['woba_value'].isna().sum()
print(f'재대결 {len(found):,}건 중 결정구를 못 찾은 건수: {missing} ({missing / len(found):.2%})')
model_data = build_avoidance_outcome_dataset(found.dropna(subset=['woba_value']))
model_data['rvae'] = compute_rvae(model_data, final_model)
dropped_total = len(found) - len(model_data)
print(f'모델링에 쓸 재대결: {len(model_data):,}건 (제외 {dropped_total}건), 투수 {model_data["pitcher"].nunique()}명')
print('포함 기준 하한 없음 확인 -- 노트북 3의 투수 439명(재대결 10건 이상)보다 많아야 함:',
      model_data['pitcher'].nunique() > 439)
display(model_data[['avoided', 'rvae', 'allowed_xbh_again']].describe())
```

- [ ] **Step 2: Run to verify expected output**

Run: `MPLBACKEND=Agg ../.venv/bin/python -u $NB_SRC/07_DY_rematch_avoidance_mixed_effects.py` (from `notebooks/`)
Expected: missing-decisive-pitch count close to notebook 3's 135/0.49% (same underlying cause — intentional walks etc.); printed pitcher count `> 439` confirms no threshold was reintroduced; no traceback.

- [ ] **Step 3: Commit**

Scratchpad source is git-ignored; note completion in the task-done ledger.

---

### Task 5: Fit model (D) — continuous RVAE

**Files:**
- Modify: `$NB_SRC/07_DY_rematch_avoidance_mixed_effects.py`

**Interfaces:**
- Consumes: `model_data` (Task 4); `fit_lmer`, `extract_lmer_fixed_effects`, `check_convergence`, `extract_variance_components` (Task 1, from `src.analysis.glmer_runner`); `compute_icc_gaussian` (Task 1, from `src.analysis.mixed_effects_model`).
- Produces: printed fixed-effects table for `avoided`, convergence status, ICC for model (D); no new module-level variables later tasks depend on (model E is independent).

- [ ] **Step 1: Write the model-fit cell**

```python
FORMULA_D = (
    'rvae ~ avoided + balls + strikes + outs_when_up + score_diff + platoon_match '
    '+ pitch_family + factor(season) + (1 | pitcher)'
)
fit_lmer(model_data, FORMULA_D)
print('=== (D) 수렴 상태 ===')
print(check_convergence())
fixed_d = extract_lmer_fixed_effects()
print('=== (D) 고정효과 (avoided 포함) ===')
display(fixed_d.round(4))
vc_d = extract_variance_components('pitcher')
icc_d = compute_icc_gaussian(vc_d['pitcher_intercept'], vc_d['residual'])
print(f'(D) ICC (투수 간 분산 비율): {icc_d:.4f} -- 노트북 3의 ω²=0.266과 비교 가능')
```

- [ ] **Step 2: Run and inspect output**

Run: `MPLBACKEND=Agg ../.venv/bin/python -u $NB_SRC/07_DY_rematch_avoidance_mixed_effects.py` (from `notebooks/`)
Expected: no traceback from the R call; `fixed_d` contains a row for `term == 'avoided'` with a finite `estimate`/`std_error`/`p_value`; `check_convergence()` output is printed verbatim (whatever it says — a singular fit is acceptable per spec, just must be visible); `icc_d` is a finite number in `[0, 1]` (or report if `extract_variance_components` returns `None` for either component — surface that explicitly rather than letting `compute_icc_gaussian` raise on `None`).

- [ ] **Step 3: Commit**

Scratchpad source is git-ignored; note completion in the task-done ledger.

---

### Task 6: Fit model (E) — binary XBH-again, and compare ICCs

**Files:**
- Modify: `$NB_SRC/07_DY_rematch_avoidance_mixed_effects.py`

**Interfaces:**
- Consumes: `model_data` (Task 4, already has `allowed_xbh_again` from Task 3); `fit_glmer`, `extract_fixed_effects_table`, `check_convergence`, `extract_variance_components` (Task 1); `compute_icc` (existing, logistic-specific, from `src.analysis.mixed_effects_model`); `icc_d` (Task 5, for the side-by-side comparison).
- Produces: printed fixed-effects table and ICC for model (E); the ω²/ICC comparison table that Task 7's interpretation references.

- [ ] **Step 1: Write the model-fit cell**

```python
FORMULA_E = (
    'allowed_xbh_again ~ avoided + balls + strikes + outs_when_up + score_diff + platoon_match '
    '+ pitch_family + factor(season) + (1 | pitcher)'
)
fit_glmer(model_data, FORMULA_E)
print('=== (E) 수렴 상태 ===')
print(check_convergence())
fixed_e = extract_fixed_effects_table()
print('=== (E) 고정효과 (avoided 포함, odds_ratio 포함) ===')
display(fixed_e.round(4))
vc_e = extract_variance_components('pitcher')
icc_e = compute_icc(vc_e['pitcher_intercept'])
print(f'(E) ICC (투수 간 분산 비율): {icc_e:.4f}')

print('\n=== ICC 비교 (투수 간 분산이 설명하는 비율) ===')
print(f'노트북 3 ω² (회피율, 편향 보정): 0.266')
print(f'(D) RVAE 모델 ICC: {icc_d:.4f}')
print(f'(E) 또-장타 모델 ICC: {icc_e:.4f}')
```

- [ ] **Step 2: Run and inspect output**

Run: `MPLBACKEND=Agg ../.venv/bin/python -u $NB_SRC/07_DY_rematch_avoidance_mixed_effects.py` (from `notebooks/`)
Expected: no traceback; `fixed_e` contains a row for `term == 'avoided'` with a finite `estimate`/`odds_ratio`/`p_value`; convergence status printed regardless of outcome; the three-line ICC comparison prints with finite numbers.

- [ ] **Step 3: Commit**

Scratchpad source is git-ignored; note completion in the task-done ledger.

---

### Task 7: Interpretation, conversion, headless execution, final verification

**Files:**
- Modify: `$NB_SRC/07_DY_rematch_avoidance_mixed_effects.py`
- Create: `notebooks/07_DY_rematch_avoidance_mixed_effects.ipynb`

**Interfaces:**
- Consumes: all prior tasks' real-data outputs (the actual printed numbers from Tasks 4–6, read from the run's output — not guessed).
- Produces: the finished, committed notebook.

- [ ] **Step 1: Write the "해석" (interpretation) markdown cell**

Using the actual numbers from the Task 4–6 runs (re-run if needed to get the exact figures), cover per spec §6: whether each model converged (and what `isSingular` says); `avoided`'s estimate/CI/p-value for both (D) and (E), stated plainly (direction and whether it's distinguishable from zero); the ICC comparison against notebook 3's ω²=0.266 and what a higher/lower ICC here would mean (if ICC here is much lower than ω², that says most of the outcome's pitcher-level clustering was already an artifact of the old aggregation, not a real pitcher effect; if similar, that cross-validates ω²); one honest sentence on what the result means for the project's running question, whichever way it comes out — per the Global Constraint against hiding a null result, a non-significant `avoided` effect here must still be reported with the same explanatory rigor as the GBM-R² explanation notebook 6 gave (why, not just that).

- [ ] **Step 2: Convert to notebook**

Run: `.venv/bin/python $NB_SRC/to_ipynb.py $NB_SRC/07_DY_rematch_avoidance_mixed_effects.py notebooks/07_DY_rematch_avoidance_mixed_effects.ipynb` (from the repo root)
Expected: `wrote notebooks/07_DY_rematch_avoidance_mixed_effects.ipynb`, exit 0.

- [ ] **Step 3: Execute headlessly**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/07_DY_rematch_avoidance_mixed_effects.ipynb` (from the repo root; no `MPLBACKEND=Agg` override needed — this notebook has no plots)
Expected: `Writing N bytes to notebooks/07_DY_rematch_avoidance_mixed_effects.ipynb`, exit 0.

- [ ] **Step 4: Verify the executed notebook has zero errors and every cell ran**

Run: a short inline Python check (`json.load` the notebook, assert no `cell['outputs']` has `output_type == 'error'`, assert no code cell's `execution_count` is `None`) — same pattern used for notebooks 3–6.
Expected: 0 error outputs; all code cells have a numeric `execution_count`.

- [ ] **Step 5: Run the full project test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, including the 2 new ones from Task 1 (86 + 2 = 88 passed, or more if other work landed meanwhile — the point is 0 failures).

- [ ] **Step 6: Scan for attribution phrases**

Run: `grep -rniE "co-authored-by|generated with" notebooks/07_DY_rematch_avoidance_mixed_effects.ipynb`
Expected: no matches (exit 1 / empty).

- [ ] **Step 7: Commit**

```bash
git add src/analysis/glmer_runner.py src/analysis/mixed_effects_model.py tests/analysis/test_mixed_effects_model.py notebooks/07_DY_rematch_avoidance_mixed_effects.ipynb
git commit -m "feat: add event-level mixed-effects check of rematch avoidance outcome"
```

(If Task 1 was already committed separately, this commit only adds the notebook.)
