"""Building blocks for the model selection-score analysis (notebook 11):
pre-game pitch usage that never looks at the game being scored, the
usage-conditional avoided-vs-reused difference with its covariance-aware
standard error, and pitcher-level resampling for a full-pipeline bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

USAGE_KEYS = ["pitcher", "season", "game_date", "pitch_type"]
DATE_KEYS = ["pitcher", "season", "game_date"]


def build_pregame_counts(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (pitcher, season, game_date, pitch_type) for every type the
    pitcher threw this season or the previous one. `n_prior`/`total_prior`
    count this season's pitches on dates strictly before `game_date`;
    `prev_share` is last season's share of that type (NaN when the pitcher
    has no previous season in the data).
    """
    p = pitches.dropna(subset=["pitch_type"])
    daily = p.groupby(USAGE_KEYS).size().rename("n_game").reset_index()
    daily_total = p.groupby(DATE_KEYS).size().rename("total_game").reset_index()
    season_type = p.groupby(["pitcher", "season", "pitch_type"]).size().rename("n_season").reset_index()
    season_total = p.groupby(["pitcher", "season"]).size().rename("total_season").reset_index()

    prev = season_type.merge(season_total, on=["pitcher", "season"])
    prev["prev_share"] = prev["n_season"] / prev["total_season"]
    prev["season"] = prev["season"] + 1
    prev_total = season_total.assign(season=season_total["season"] + 1, has_prev_season=True)

    types = pd.concat([
        season_type[["pitcher", "season", "pitch_type"]],
        prev[["pitcher", "season", "pitch_type"]].merge(season_total[["pitcher", "season"]], on=["pitcher", "season"]),
    ]).drop_duplicates()

    daily_total = daily_total.sort_values("game_date")
    daily_total["total_prior"] = (
        daily_total.groupby(["pitcher", "season"])["total_game"].cumsum() - daily_total["total_game"]
    )

    grid = daily_total.merge(types, on=["pitcher", "season"])
    grid = grid.merge(daily, on=USAGE_KEYS, how="left")
    grid["n_game"] = grid["n_game"].fillna(0).astype(int)
    grid = grid.sort_values("game_date")
    grid["n_prior"] = grid.groupby(["pitcher", "season", "pitch_type"])["n_game"].cumsum() - grid["n_game"]

    grid = grid.merge(prev_total[["pitcher", "season", "has_prev_season"]], on=["pitcher", "season"], how="left")
    grid["has_prev_season"] = grid["has_prev_season"].fillna(False).astype(bool)
    grid = grid.merge(prev[["pitcher", "season", "pitch_type", "prev_share"]], on=["pitcher", "season", "pitch_type"], how="left")
    grid.loc[grid["has_prev_season"] & grid["prev_share"].isna(), "prev_share"] = 0.0
    return grid.reset_index(drop=True)


def smoothed_usage(grid: pd.DataFrame, smoothing_k: float, min_prior_pitches: int) -> pd.Series:
    """Pre-game usage share. With a previous season it is the season-to-date
    share shrunk toward last season's share by `smoothing_k` pseudo-pitches;
    without one it is the raw season-to-date share once `min_prior_pitches`
    have been thrown; otherwise it stays missing.
    """
    shrunk = (grid["n_prior"] + smoothing_k * grid["prev_share"]) / (grid["total_prior"] + smoothing_k)
    raw = grid["n_prior"] / grid["total_prior"].where(grid["total_prior"] >= min_prior_pitches)
    return shrunk.where(grid["has_prev_season"], raw)


def lookup_pregame_usage(queries: pd.DataFrame, grid: pd.DataFrame, min_prior_pitches: int) -> pd.Series:
    """Usage for each query row (USAGE_KEYS). A type absent from the pitcher's
    history is a real 0 when that history exists, and missing when it does not.
    """
    merged = queries[USAGE_KEYS].merge(grid[[*USAGE_KEYS, "usage"]], on=USAGE_KEYS, how="left")
    info = grid.drop_duplicates(DATE_KEYS)[[*DATE_KEYS, "has_prev_season", "total_prior"]]
    merged = merged.merge(info, on=DATE_KEYS, how="left")
    has_history = merged["has_prev_season"].fillna(False).astype(bool) | (merged["total_prior"] >= min_prior_pitches)
    type_known = queries[USAGE_KEYS].merge(
        grid[USAGE_KEYS].assign(_known=True), on=USAGE_KEYS, how="left"
    )["_known"].notna()
    usage = merged["usage"].where(type_known.to_numpy(), np.where(has_history, 0.0, np.nan))
    return pd.Series(usage.to_numpy(), index=queries.index)


def tune_smoothing_k(grid: pd.DataFrame, seasons, candidates, min_prior_pitches: int):
    """Pick the shrinkage strength that best predicts each game's actual usage
    share from pre-game information, using only `seasons`.
    """
    rows = grid[grid["season"].isin(list(seasons)) & grid["has_prev_season"] & (grid["total_game"] > 0)]
    actual = rows["n_game"] / rows["total_game"]
    records = []
    for k in candidates:
        err = (actual - smoothed_usage(rows, k, min_prior_pitches)) ** 2
        records.append({"smoothing_k": k, "weighted_mse": float(np.average(err, weights=rows["total_game"]))})
    table = pd.DataFrame(records)
    return table.sort_values("weighted_mse").iloc[0]["smoothing_k"], table


def conditional_difference(usage_centered, beta_main, beta_interaction, var_main, var_interaction, cov_main_interaction):
    """Delta(u) = beta_main + beta_interaction * u and its standard error,
    covariance between the two coefficients included.
    """
    u = np.asarray(usage_centered, dtype=float)
    delta = beta_main + beta_interaction * u
    variance = var_main + u ** 2 * var_interaction + 2 * u * cov_main_interaction
    return delta, np.sqrt(variance)


def standardized_group_curves(coef: pd.Series, vcov: pd.DataFrame, design_means: pd.Series, usage_centered,
                              avoided_term: str, usage_term: str, interaction_term: str) -> pd.DataFrame:
    """Fixed-effect predictions for each group along the usage grid, with every
    other covariate held at the sample's design-matrix mean (random effects at 0).
    """
    terms = list(coef.index)
    beta, v = coef.to_numpy(), vcov.loc[terms, terms].to_numpy()
    records = []
    for avoided in (0, 1):
        for u in np.asarray(usage_centered, dtype=float):
            x = design_means.loc[terms].copy()
            x[avoided_term], x[usage_term], x[interaction_term] = avoided, u, avoided * u
            xv = x.to_numpy(dtype=float)
            records.append({"avoided": avoided, "usage_centered": u, "predicted": float(xv @ beta),
                            "se": float(np.sqrt(xv @ v @ xv))})
    return pd.DataFrame(records)


def common_support(usage_avoided, usage_reused, lower_q: float = 0.025, upper_q: float = 0.975):
    """Usage range where both groups have observations: the overlap of each
    group's [lower_q, upper_q] quantile range.
    """
    lo = max(np.quantile(usage_avoided, lower_q), np.quantile(usage_reused, lower_q))
    hi = min(np.quantile(usage_avoided, upper_q), np.quantile(usage_reused, upper_q))
    return float(lo), float(hi)


def expand_by_pitcher(index_map: dict, sampled_pitchers) -> tuple[np.ndarray, np.ndarray]:
    """Row positions for a with-replacement draw of pitchers, plus a replicate
    id per row so a pitcher drawn twice becomes two separate clusters.
    """
    rows, replicate = [], []
    for draw, pitcher in enumerate(sampled_pitchers):
        idx = index_map.get(pitcher)
        if idx is None or len(idx) == 0:
            continue
        rows.append(idx)
        replicate.append(np.full(len(idx), draw))
    if not rows:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(rows), np.concatenate(replicate)


def selection_score_parts(df: pd.DataFrame, model, features, cf_type_col: str, cf_usage_col: str | None,
                          usage_feature: str | None):
    """(prediction if the hit pitch type were thrown, prediction for the pitch
    actually thrown). The candidate type's usage is swapped in with the type.
    """
    if usage_feature is not None and cf_usage_col is None:
        raise ValueError("usage_feature is a model input, so the candidate type's usage column is required")
    actual = np.asarray(model.predict(df[features]))
    cf = df[features].copy()
    cf["pitch_type"] = df[cf_type_col].astype(df["pitch_type"].dtype)
    if usage_feature is not None:
        cf[usage_feature] = df[cf_usage_col].to_numpy()
    return np.asarray(model.predict(cf)), actual
