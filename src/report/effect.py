"""Step 3 of the final report (p.8-10): was avoiding the hit type a good
choice, judged by the GBM selection score.

p.9  -- mean selection score, avoided vs reused rematches (all hit types).
p.10 -- whether that gap depends on how much the pitcher uses the hit type
        (non-fastball hits only), under the original specification and under
        the re-validation (pre-game usage, GBM trained only on earlier seasons).

Mixed models are fitted with R's lme4 through rpy2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from src.analysis.glmer_runner import (
    check_convergence,
    extract_fixed_effects_vcov,
    extract_lmer_fixed_effects,
    fit_lmer,
)
from src.analysis.selection_score import (
    build_pregame_counts,
    common_support,
    conditional_difference,
    expand_by_pitcher,
    lookup_pregame_usage,
    smoothed_usage,
    tune_smoothing_k,
)
from src.analysis.selection_value import (
    COV_DECISIVE,
    COV_START,
    FEATURES_NO_USAGE,
    FEATURES_PREGAME,
    FEATURES_SEASON,
    PA_KEYS,
    RANDOM_STATE,
    SCORE_COLUMNS,
    build_decisive_table,
    build_pa_pitches,
    build_rematch_events,
    fit_gbm,
    permutation_test_gap,
    refit_gap_draws,
    score_pa,
)

TRAIN_SEASONS = (2021, 2022, 2023, 2024, 2025)
OOT_FOLD_SEASONS = (2022, 2023, 2024, 2025, 2026)   # each season is scored by a GBM trained on the seasons before it
PRIMARY_EVAL_SEASON = 2026
MIN_PRIOR_PITCHES = 100
SMOOTHING_K_CANDIDATES = (5, 10, 25, 50, 100, 200, 400, 800)


def fit_mixed(data: pd.DataFrame, outcome: str, usage_c: str | None, covars: list[str], season_fe: bool = True) -> dict:
    """lmer of `outcome` on avoided (x centered usage when `usage_c` is given),
    the situation covariates, season fixed effects and a pitcher random intercept.
    """
    rhs = f"avoided * {usage_c}" if usage_c else "avoided"
    formula = f"{outcome} ~ {rhs} + {' + '.join(covars)}{' + factor(season)' if season_fe else ''} + (1 | pitcher_id)"
    cols = [outcome, "avoided", *([usage_c] if usage_c else []), *covars, "season", "pitcher_id"]
    fit_lmer(data[cols], formula)
    return {"formula": formula, "coef": extract_lmer_fixed_effects().set_index("term"),
            "vcov": extract_fixed_effects_vcov(), "convergence": check_convergence(),
            "n": len(data), "n_pitchers": int(data["pitcher_id"].nunique())}


def delta_at(fit: dict, usage_raw, center: float, usage_c: str):
    """Avoided - reused difference at the given (uncentered) usage values, with its standard error."""
    c, v, term = fit["coef"], fit["vcov"], f"avoided:{usage_c}"
    return conditional_difference(np.asarray(usage_raw, dtype=float) - center, c.loc["avoided", "estimate"],
                                  c.loc[term, "estimate"], v.loc["avoided", "avoided"], v.loc[term, term],
                                  v.loc["avoided", term])


def interaction_row(label: str, fit: dict, usage_c: str) -> dict:
    term = fit["coef"].loc[f"avoided:{usage_c}"]
    return {"spec": label, "n": fit["n"], "n_pitchers": fit["n_pitchers"], "interaction": term["estimate"],
            "ci_low": term["ci_low"], "ci_high": term["ci_high"], "ci_kind": "GBM 고정"}


def prepare(pitches: pd.DataFrame, season_usage: pd.DataFrame, xbh: pd.DataFrame) -> dict:
    """Training table, rematch events and their pitches, scored with the
    original specification (GBM on every season, season-wide usage).
    """
    decisive = build_decisive_table(pitches, season_usage)
    events = build_rematch_events(xbh, decisive, season_usage, pitches)
    pa = build_pa_pitches(events, pitches, season_usage, decisive)
    train = decisive[decisive["season"].isin(TRAIN_SEASONS)]
    valid = decisive[decisive["season"] == PRIMARY_EVAL_SEASON]
    validation_r2 = r2_score(valid["woba_value"], fit_gbm(train, FEATURES_SEASON).predict(valid[FEATURES_SEASON]))
    gbm = fit_gbm(decisive, FEATURES_SEASON)
    scores = score_pa(pa, gbm, FEATURES_SEASON, "hit_usage_season", "season_usage_rate")
    events = events.merge(scores, on=PA_KEYS, how="left", validate="one_to_one")
    return {"decisive": decisive, "events": events, "pa": pa, "validation_r2": float(validation_r2)}


def selection_score_gap(ctx: dict, n_refit: int, n_perm: int) -> dict:
    """p.9: avoided - reused mean selection score over all rematches, with the
    lmer SE, the spread over `n_refit` GBM refits, their root-sum-square as a
    conservative SE, and a permutation test.
    """
    events = ctx["events"]
    fit = fit_mixed(events, "selection_score", None, COV_DECISIVE)
    avoided = fit["coef"].loc["avoided"]
    out = {"n": fit["n"], "n_pitchers": fit["n_pitchers"], "estimate": float(avoided["estimate"]),
           "lmer_se": float(avoided["std_error"]), "convergence": fit["convergence"], "n_refit": n_refit,
           "n_perm": n_perm}
    if n_refit >= 2:
        draws = refit_gap_draws(ctx["decisive"], ctx["pa"], events, n_refit, RANDOM_STATE)
        out["refit_sd"] = float(np.std(draws))
        se = float(np.sqrt(out["lmer_se"] ** 2 + out["refit_sd"] ** 2))
        out.update({"conservative_se": se, "ci_low": out["estimate"] - 1.96 * se, "ci_high": out["estimate"] + 1.96 * se,
                    "p_value": float(2 * stats.norm.sf(abs(out["estimate"]) / se))})
    if n_perm > 0:
        perm = permutation_test_gap(events["selection_score"].to_numpy(), events["avoided"].to_numpy(), n_perm, RANDOM_STATE)
        out.update({"raw_gap": perm["observed_gap"], "permutation_p": perm["p_value"]})
    return out


def usage_interaction(ctx: dict, pitches: pd.DataFrame, n_boot: int) -> dict:
    """p.10: the avoided x hit-type-usage interaction on non-fastball hits under
    (1) the original specification, (2) pre-game usage with out-of-time GBMs,
    pooled over 2022-2026, and (3) 2021-25 training -> 2026 evaluation only.
    With `n_boot` > 0, (2) and (3) also get a full-pipeline bootstrap interval
    (resample pitchers -> refit the GBMs -> rescore -> refit the same lmer).
    """
    decisive, events, pa = ctx["decisive"], ctx["events"], ctx["pa"]
    nonfb = events[events["hit_pitch_family"] != "fastball"].copy()
    center_season = float(nonfb["hit_usage_season"].mean())
    nonfb["hit_usage_season_c"] = nonfb["hit_usage_season"] - center_season
    fit_base = fit_mixed(nonfb, "selection_score", "hit_usage_season_c", COV_DECISIVE)
    support = common_support(nonfb.loc[nonfb["avoided"] == 1, "hit_usage_season"],
                             nonfb.loc[nonfb["avoided"] == 0, "hit_usage_season"])

    # dropping the usage input from the GBM, everything else as in the original specification
    gbm_no_usage = fit_gbm(decisive, FEATURES_NO_USAGE)
    no_usage = nonfb.drop(columns=SCORE_COLUMNS).merge(score_pa(pa, gbm_no_usage, FEATURES_NO_USAGE, None, None), on=PA_KEYS)
    fit_no_usage = fit_mixed(no_usage, "selection_score", "hit_usage_season_c", COV_DECISIVE)

    # pre-game usage: season-to-date share shrunk toward last season's share; k is tuned without the 2026 season
    grid = build_pregame_counts(pitches)
    smoothing_k, _ = tune_smoothing_k(grid, seasons=range(2022, PRIMARY_EVAL_SEASON),
                                      candidates=SMOOTHING_K_CANDIDATES, min_prior_pitches=MIN_PRIOR_PITCHES)
    grid["usage"] = smoothed_usage(grid, smoothing_k, MIN_PRIOR_PITCHES)

    def pregame_usage_of(df: pd.DataFrame, type_col: str) -> np.ndarray:
        queries = df[["pitcher", "season", "game_date"]].assign(pitch_type=df[type_col].astype(object).to_numpy())
        return lookup_pregame_usage(queries, grid, MIN_PRIOR_PITCHES).to_numpy()

    decisive = decisive.assign(pregame_usage=pregame_usage_of(decisive, "pitch_type"))
    decisive_pre = decisive.dropna(subset=["pregame_usage"]).reset_index(drop=True)
    events = events.assign(hit_usage_pregame=pregame_usage_of(events, "hit_pitch_type"))
    pa = pa.assign(pregame_usage=pregame_usage_of(pa, "pitch_type"), hit_usage_pregame=pregame_usage_of(pa, "hit_pitch_type"))
    x_pre, y_pre, season_pre = decisive_pre[FEATURES_PREGAME], decisive_pre["woba_value"].to_numpy(), decisive_pre["season"].to_numpy()
    pa_pre = pa.dropna(subset=["pregame_usage", "hit_usage_pregame"])
    pa_by_season = {y: pa_pre[pa_pre["season"] == y] for y in OOT_FOLD_SEASONS}

    fold_models = {y: fit_gbm(decisive_pre[season_pre < y], FEATURES_PREGAME) for y in OOT_FOLD_SEASONS}
    oot_scores = pd.concat([score_pa(pa_by_season[y], fold_models[y], FEATURES_PREGAME, "hit_usage_pregame", "pregame_usage")
                            for y in OOT_FOLD_SEASONS], ignore_index=True)
    oot = events.drop(columns=SCORE_COLUMNS).merge(oot_scores, on=PA_KEYS, how="inner", validate="one_to_one")
    oot = oot.dropna(subset=["hit_usage_pregame", *COV_START])
    oot_nonfb = oot[oot["hit_pitch_family"] != "fastball"].copy()
    center_pre = float(oot_nonfb["hit_usage_pregame"].mean())
    oot_nonfb["hit_usage_pregame_c"] = oot_nonfb["hit_usage_pregame"] - center_pre
    oot_primary = oot_nonfb[oot_nonfb["season"] == PRIMARY_EVAL_SEASON]
    fit_pooled = fit_mixed(oot_nonfb, "selection_score", "hit_usage_pregame_c", COV_START, season_fe=True)
    fit_primary = fit_mixed(oot_primary, "selection_score", "hit_usage_pregame_c", COV_START, season_fe=False)

    rows = [
        interaction_row("기존: 시즌 전체 구사율, 전체 시즌 학습", fit_base, "hit_usage_season_c"),
        interaction_row("경기 전 구사율, 시간 외 합산 평가", fit_pooled, "hit_usage_pregame_c"),
        interaction_row("2021–25 학습 → 2026 평가", fit_primary, "hit_usage_pregame_c"),
    ]
    boot = pd.DataFrame()
    if n_boot >= 2:
        boot_events = {y: oot_nonfb[oot_nonfb["season"] == y].reset_index(drop=True) for y in OOT_FOLD_SEASONS}
        boot_pa = {y: pa_by_season[y].merge(boot_events[y][PA_KEYS], on=PA_KEYS) for y in OOT_FOLD_SEASONS}
        boot = _full_bootstrap(
            n_boot, RANDOM_STATE,
            pitcher_pool=np.array(sorted(set(decisive["pitcher"]) | set(nonfb["pitcher"]))),
            train_index=decisive_pre.groupby("pitcher").indices, x_train=x_pre, y_train=y_pre, season_train=season_pre,
            events_by_season=boot_events, pa_by_season=boot_pa,
        )
        for row, name in ((rows[1], "pooled"), (rows[2], "primary")):
            draws = boot[name].dropna().to_numpy()
            if len(draws) >= 2:
                row["ci_low"], row["ci_high"] = np.percentile(draws, [2.5, 97.5])
                row["ci_kind"] = f"전체 과정 부트스트랩 ({len(draws)}회)"
    grid_u = np.linspace(0.0, float(np.ceil(nonfb["hit_usage_season"].quantile(0.999) * 20) / 20), 200)
    delta, delta_se = delta_at(fit_base, grid_u, center_season, "hit_usage_season_c")
    curve = pd.DataFrame({"usage": grid_u, "delta": delta, "se": delta_se,
                          "in_support": (grid_u >= support[0]) & (grid_u <= support[1])})
    return {"table": pd.DataFrame(rows), "no_usage_interaction": float(fit_no_usage["coef"].loc["avoided:hit_usage_season_c", "estimate"]),
            "smoothing_k": float(smoothing_k), "support": support, "curve": curve, "bootstrap": boot,
            "usage_by_group": nonfb[["avoided", "hit_usage_season"]]}


def _full_bootstrap(n_boot, seed, pitcher_pool, train_index, x_train, y_train, season_train, events_by_season, pa_by_season):
    """Interaction coefficient of the pooled and the 2026-only out-of-time
    models over `n_boot` pitcher-level resamples. A pitcher drawn twice
    becomes two clusters; a failed lmer fit is left missing, not dropped silently.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed)
    event_index = {y: events_by_season[y].groupby("pitcher").indices for y in OOT_FOLD_SEASONS}
    columns = [*PA_KEYS, "avoided", "hit_usage_pregame_c", *COV_START, "season"]
    records = []
    for b in range(n_boot):
        sampled = rng.choice(pitcher_pool, size=len(pitcher_pool), replace=True)
        train_rows, _ = expand_by_pitcher(train_index, sampled)
        train_season = season_train[train_rows]
        parts = []
        for y in OOT_FOLD_SEASONS:
            rows = train_rows[train_season < y]
            model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=seed + b, max_iter=200,
                                                  early_stopping=True).fit(x_train.iloc[rows], y_train[rows])
            scored = events_by_season[y][columns].merge(
                score_pa(pa_by_season[y], model, FEATURES_PREGAME, "hit_usage_pregame", "pregame_usage"), on=PA_KEYS)
            event_rows, replicate = expand_by_pitcher(event_index[y], sampled)
            parts.append(scored.iloc[event_rows].assign(pitcher_id=[f"r{r}" for r in replicate]))
        sample = pd.concat(parts, ignore_index=True)
        record = {"replicate": b, "pooled": np.nan, "primary": np.nan}
        for name, data, season_fe in (("pooled", sample, True),
                                      ("primary", sample[sample["season"] == PRIMARY_EVAL_SEASON], False)):
            try:
                fit = fit_mixed(data, "selection_score", "hit_usage_pregame_c", COV_START, season_fe)
                record[name] = fit["coef"].loc["avoided:hit_usage_pregame_c", "estimate"]
            except Exception:
                pass
        records.append(record)
    return pd.DataFrame(records)
