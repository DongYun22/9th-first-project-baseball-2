"""Data assembly for the selection-score analysis (final report p.9-10).

A GBM predicts the wOBA value of a PA-ending pitch from the situation, the
pitch type and that type's usage. Every pitch of a rematch PA is then
predicted twice -- as thrown, and with the pitch type swapped to the one
that was hit for the extra-base hit -- and the PA's selection score is the
mean of (prediction if the hit type were reused - prediction as thrown).
A positive score means the model rates the actual choice above reusing the
hit type.

The swap itself and the pre-game usage helpers live in selection_score.py;
the mixed models are fitted in src/report/effect.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.analysis.selection_score import selection_score_parts
from src.preprocessing.pitch_family import map_pitch_family

RANDOM_STATE = 20261003
CONTEXT_FEATURES = ["stand", "p_throws", "balls", "strikes", "outs_when_up", "runners_n", "risp", "score_diff"]
FEATURES_SEASON = ["pitch_type", *CONTEXT_FEATURES, "season_usage_rate"]
FEATURES_PREGAME = ["pitch_type", *CONTEXT_FEATURES, "pregame_usage"]
FEATURES_NO_USAGE = ["pitch_type", *CONTEXT_FEATURES]
CATEGORICAL = ["pitch_type", "stand", "p_throws"]
COV_DECISIVE = ["outs_when_up", "score_diff", "platoon_match", "risp", "runners_n"]
COV_START = [f"{c}_start" for c in COV_DECISIVE]
PA_KEYS = ["game_pk", "at_bat_number"]
SCORE_COLUMNS = ["selection_score", "pred_reuse", "pred_actual"]
USAGE_KEYS = ["pitcher", "season", "pitch_type"]


def add_platoon_match(pitches: pd.DataFrame) -> pd.DataFrame:
    """1.0 when batter and pitcher share a hand, 0.0 when not, missing when either is unknown."""
    out = pitches.copy()
    out["platoon_match"] = np.where(
        out["stand"].isna() | out["p_throws"].isna(), np.nan, (out["stand"] == out["p_throws"]).astype(float)
    )
    return out


def build_decisive_table(pitches: pd.DataFrame, season_usage: pd.DataFrame) -> pd.DataFrame:
    """PA-ending pitches (the ones carrying `woba_value`) with their type's
    season usage -- the GBM's training rows.
    """
    decisive = pitches[pitches["woba_value"].notna() & pitches["pitch_type"].notna()].merge(
        season_usage[[*USAGE_KEYS, "season_usage_rate"]], on=USAGE_KEYS, how="left"
    )
    for col in CATEGORICAL:
        decisive[col] = decisive[col].astype("category")
    return decisive


def fit_gbm(table: pd.DataFrame, features: list[str], seed: int = RANDOM_STATE) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        categorical_features="from_dtype", random_state=seed, max_iter=200, early_stopping=True
    )
    return model.fit(table[features], table["woba_value"])


def build_rematch_events(
    xbh: pd.DataFrame, decisive_table: pd.DataFrame, season_usage: pd.DataFrame, pitches: pd.DataFrame
) -> pd.DataFrame:
    """One row per rematch PA that has an ending pitch and every model input:
    the ending pitch's situation, `avoided` (1 = the hit type was not thrown
    in the PA), the hit type's season usage, and the PA's starting situation
    (`*_start`, from its first pitch).
    """
    rematch = xbh[["game_pk", "pitcher", "season", "game_date", "next_at_bat_number", "hit_pitch_type",
                   "reused_same_type"]].rename(columns={"next_at_bat_number": "at_bat_number"})
    rematch["at_bat_number"] = rematch["at_bat_number"].astype(int)
    rematch["game_date"] = pd.to_datetime(rematch["game_date"])
    ending = decisive_table[["game_pk", "pitcher", "at_bat_number", "pitch_type", *CONTEXT_FEATURES,
                             "season_usage_rate", "platoon_match", "events", "woba_value"]]
    events = rematch.merge(ending, on=["game_pk", "pitcher", "at_bat_number"], how="left", validate="many_to_one")
    events["ending_pitch_family"] = events["pitch_type"].map(map_pitch_family)
    events = events.dropna(subset=["woba_value", "events", "platoon_match", "pitch_type", *CONTEXT_FEATURES,
                                   "season_usage_rate", "ending_pitch_family", "reused_same_type"]).copy()
    events["avoided"] = (1 - events["reused_same_type"]).astype(int)
    events["pitcher_id"] = events["pitcher"].astype(str)
    events["hit_pitch_family"] = events["hit_pitch_type"].map(map_pitch_family)
    first_pitch = pitches.sort_values("pitch_number").drop_duplicates(PA_KEYS)[[*PA_KEYS, *COV_DECISIVE]]
    events = events.merge(first_pitch.rename(columns=dict(zip(COV_DECISIVE, COV_START))), on=PA_KEYS, how="left")
    hit_usage = season_usage.rename(columns={"pitch_type": "hit_pitch_type", "season_usage_rate": "hit_usage_season"})
    events = events.merge(hit_usage[["pitcher", "season", "hit_pitch_type", "hit_usage_season"]],
                          on=["pitcher", "season", "hit_pitch_type"], how="left")
    events["hit_usage_season"] = events["hit_usage_season"].fillna(0.0)
    return events


def build_pa_pitches(
    events: pd.DataFrame, pitches: pd.DataFrame, season_usage: pd.DataFrame, decisive_table: pd.DataFrame
) -> pd.DataFrame:
    """Every typed pitch of the rematch PAs, with the categorical dtypes the GBM was trained on."""
    pa = events[[*PA_KEYS, "pitcher", "season", "game_date", "hit_pitch_type", "hit_usage_season"]].merge(
        pitches[[*PA_KEYS, "pitch_number", "pitch_type", *CONTEXT_FEATURES]], on=PA_KEYS, how="left"
    ).dropna(subset=["pitch_type"])
    pa = pa.merge(season_usage[[*USAGE_KEYS, "season_usage_rate"]], on=USAGE_KEYS, how="left").dropna(
        subset=["season_usage_rate"]
    )
    for col in CATEGORICAL:
        pa[col] = pa[col].astype(decisive_table[col].dtype)
    return pa.reset_index(drop=True)


def score_pa(pa: pd.DataFrame, model, features: list[str], cf_usage_col: str | None,
             usage_feature: str | None) -> pd.DataFrame:
    """Per PA: mean selection score and its two parts (prediction if the hit
    type were reused, prediction for the pitch actually thrown).
    """
    cf, actual = selection_score_parts(pa, model, features, "hit_pitch_type", cf_usage_col, usage_feature)
    out = pa[PA_KEYS].assign(selection_score=cf - actual, pred_reuse=cf, pred_actual=actual)
    return out.groupby(PA_KEYS, sort=False)[SCORE_COLUMNS].mean().reset_index()


def group_gap(values, avoided) -> float:
    """Mean of `values` in the avoided group minus the reused group."""
    values, avoided = np.asarray(values, dtype=float), np.asarray(avoided)
    return float(values[avoided == 1].mean() - values[avoided == 0].mean())


def permutation_test_gap(values, avoided, n_perm: int, seed: int) -> dict:
    """Two-sided permutation test of the avoided - reused mean gap."""
    rng = np.random.default_rng(seed)
    values, avoided = np.asarray(values, dtype=float), np.asarray(avoided)
    observed = group_gap(values, avoided)
    n, n1 = len(values), int(avoided.sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        order = rng.permutation(n)
        null[i] = values[order[:n1]].mean() - values[order[n1:]].mean()
    return {"observed_gap": observed, "p_value": float((np.abs(null) >= abs(observed)).mean()),
            "null_sd": float(null.std())}


def refit_gap_draws(train: pd.DataFrame, pa: pd.DataFrame, events: pd.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    """Avoided - reused gap of the PA selection score under `n_boot` GBMs,
    each refitted on a row-level resample of `train`. Its spread is the part
    of the uncertainty that fixing the GBM hides.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sample = train.iloc[rng.integers(0, len(train), size=len(train))]
        model = fit_gbm(sample, FEATURES_SEASON, seed + b)
        scored = score_pa(pa, model, FEATURES_SEASON, "hit_usage_season", "season_usage_rate")
        merged = events[[*PA_KEYS, "avoided"]].merge(scored, on=PA_KEYS, how="inner")
        draws[b] = group_gap(merged["selection_score"], merged["avoided"])
    return draws
