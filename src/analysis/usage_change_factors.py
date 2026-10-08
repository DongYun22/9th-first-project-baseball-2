"""Which features of an extra-base hit go with a larger drop in the hit
pitch type's usage (final report p.7; Beomseok's GBM + SHAP analysis).

Unlike the rest of the report, the rematch here is the next PA against the
same batter anywhere in the same season, not only in the same game.
usage_change = share of the hit type in that PA - the pitcher's season
usage of the type (negative = avoided). SHAP shows which inputs the model
leans on; it is not a causal effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

XBH_EVENTS = ["double", "triple", "home_run"]
MAIN_PITCH_TYPES = ["FF", "SI", "SL", "CH", "FC", "CU", "ST", "FS", "KC"]
NUMERIC_FEATURES = [
    "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
    "release_speed", "pfx_x_arm", "pfx_z", "plate_x", "plate_z",
    "runner_1b", "runner_2b", "runner_3b", "score_diff_bat",
]
CATEGORICAL_FEATURES = ["events", "pitch_group", "matchup", "count_state", "out_state"]
FEATURES = [
    "events", "launch_speed", "launch_angle", "estimated_woba_using_speedangle", "pitch_group",
    "release_speed", "pfx_x_arm", "pfx_z", "plate_x", "plate_z", "matchup", "count_state", "out_state",
    "runner_1b", "runner_2b", "runner_3b", "score_diff_bat",
]
TARGET = "usage_change"
SEED = 42
PA_ORDER = ["season", "pitcher", "batter", "game_date", "game_pk", "at_bat_number"]


def build_same_season_rematch_events(pitches: pd.DataFrame, season_usage: pd.DataFrame) -> pd.DataFrame:
    """XBH pitches whose pitcher faces the same batter again later that season,
    with the hit type's share in that next PA (`post_usage`), its season usage
    (`baseline_usage`) and their difference (`usage_change`). Events whose
    next PA has no typed pitch, or whose hit type is unknown, are left out.
    """
    pas = pitches[PA_ORDER].drop_duplicates().sort_values(PA_ORDER)
    matchup = pas.groupby(["season", "pitcher", "batter"])
    pas["next_game_pk"] = matchup["game_pk"].shift(-1)
    pas["next_at_bat_number"] = matchup["at_bat_number"].shift(-1)
    xbh = pitches[pitches["events"].isin(XBH_EVENTS) & pitches["pitch_type"].notna()]
    events = xbh.merge(pas[["game_pk", "at_bat_number", "pitcher", "batter", "next_game_pk", "next_at_bat_number"]],
                       on=["game_pk", "at_bat_number", "pitcher", "batter"])
    events = events[events["next_game_pk"].notna()]

    typed = pitches[pitches["pitch_type"].notna()]
    pa_keys = ["game_pk", "at_bat_number", "pitcher"]
    total = typed.groupby(pa_keys).size().rename("total_post_pitches").reset_index()
    by_type = typed.groupby([*pa_keys, "pitch_type"]).size().rename("post_x_count").reset_index()
    rename = {"game_pk": "next_game_pk", "at_bat_number": "next_at_bat_number"}
    events = events.merge(total.rename(columns=rename), on=["next_game_pk", "next_at_bat_number", "pitcher"], how="inner")
    events = events.merge(by_type.rename(columns=rename), on=["next_game_pk", "next_at_bat_number", "pitcher", "pitch_type"],
                          how="left")
    events["post_usage"] = events["post_x_count"].fillna(0) / events["total_post_pitches"]
    usage = season_usage.rename(columns={"season_usage_rate": "baseline_usage"})
    events = events.merge(usage[["pitcher", "season", "pitch_type", "baseline_usage"]], on=["pitcher", "season", "pitch_type"])
    events[TARGET] = events["post_usage"] - events["baseline_usage"]
    return events.reset_index(drop=True)


def build_design_matrix(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs of the final specification, one-hot encoded with the first
    level of each categorical dropped; rows missing any input are removed.
    `score_diff` must be from the pitching team's side; it is flipped to the
    batting team's (`score_diff_bat`, + = pitcher's team trailing).
    """
    d = events.copy()
    d["score_diff_bat"] = -d["score_diff"]
    d["pfx_x_arm"] = np.where(d["p_throws"] == "R", d["pfx_x"], -d["pfx_x"])
    for base in ("1b", "2b", "3b"):
        d[f"runner_{base}"] = d[f"on_{base}"].notna().astype(int)
    d["pitch_group"] = d["pitch_type"].where(d["pitch_type"].isin(MAIN_PITCH_TYPES), "Other")
    d = d.dropna(subset=["p_throws", "stand", "balls", "strikes", "outs_when_up"])
    d["matchup"] = d["p_throws"].astype(str) + "-" + d["stand"].astype(str)
    d["count_state"] = d["balls"].astype(int).astype(str) + "-" + d["strikes"].astype(int).astype(str)
    d["out_state"] = d["outs_when_up"].astype(int).astype(str) + "_out"
    d = d[[*FEATURES, TARGET]].dropna()
    X = pd.get_dummies(d[FEATURES], columns=CATEGORICAL_FEATURES, drop_first=True)
    return X, d[TARGET]


def fit_and_explain(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit the GBM on a random 80% of the events and compute mean |SHAP| per
    input on the held-out 20%. Returns the test R² and the importance table.
    """
    import shap

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=SEED).fit(X_train, y_train)
    values = shap.Explainer(model)(X_test).values
    importance = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    return {"n": len(X), "n_test": len(X_test), "r2": float(r2_score(y_test, model.predict(X_test))),
            "importance": importance}
