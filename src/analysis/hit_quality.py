"""Hit quality (xwOBA, barrel) vs. pitch avoidance, for every batted-ball outcome.

Every batted ball (field_out, single, double/triple, home_run) with a comparison PA becomes an
event. Its avoidance is `diff = baseline - share of the event pitch type in the comparison PA`,
the same quantity as the main analysis (`build_event_dataset_for_events`), but computed with
vectorized group counts because there are hundreds of thousands of events.

Reference group = weakly hit outs (field_out in the lowest xwOBA quartile): the pitcher "won"
both on result and on contact quality. Each outcome x xwOBA-quartile cell is compared to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.situation_matching import ELIGIBILITY
from src.preprocessing.pitch_family import map_pitch_family

OUTCOME_GROUPS = {
    "field_out": "아웃", "single": "단타", "double": "2·3루타", "triple": "2·3루타", "home_run": "홈런",
}
OUTCOME_ORDER = ["아웃", "단타", "2·3루타", "홈런"]
XWOBA = "estimated_woba_using_speedangle"


def is_barrel(launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    """Statcast barrel approximation (baseballr): EV >= 98, LA <= 50, 1.5*EV - LA >= 117, EV + LA >= 124."""
    ev, la = launch_speed, launch_angle
    out = (ev >= 98) & (la <= 50) & (1.5 * ev - la >= 117) & (ev + la >= 124)
    return out.where(ev.notna() & la.notna())


def add_prior_game_usage(pitches: pd.DataFrame) -> pd.DataFrame:
    """`baseline_usage`: share of this pitch's type among the pitcher's earlier pitches in the game
    (pitches with a missing type count in the denominator, as in compute_baseline_usage)."""
    out = pitches.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"]).copy()
    n_prior = out.groupby(["game_pk", "pitcher"]).cumcount()
    n_prior_type = out.groupby(["game_pk", "pitcher", "pitch_type"]).cumcount()
    out["baseline_usage"] = (n_prior_type / n_prior).where((n_prior > 0) & out["pitch_type"].notna())
    return out


def comparison_at_bat(pitches: pd.DataFrame, events: pd.DataFrame, mode: str) -> pd.Series:
    """at_bat_number of each event's comparison PA (`mode` as in situation_matching)."""
    if mode == "next_batter":
        return events["at_bat_number"] + 1
    pa = pitches[["game_pk", "batter", "at_bat_number"]].drop_duplicates().sort_values(
        ["game_pk", "batter", "at_bat_number"]
    )
    pa["next_ab"] = pa.groupby(["game_pk", "batter"])["at_bat_number"].shift(-1)
    return events.merge(pa, on=["game_pk", "batter", "at_bat_number"], how="left")["next_ab"].set_axis(events.index)


def comparison_share(pitches: pd.DataFrame, events: pd.DataFrame, next_ab: pd.Series) -> pd.DataFrame:
    """Pitch count of the comparison PA (same pitcher) and the share of the event pitch type in it."""
    n_all = pitches.groupby(["game_pk", "at_bat_number", "pitcher"]).size().rename("next_ab_pitch_count")
    n_type = pitches.groupby(["game_pk", "at_bat_number", "pitcher", "pitch_type"]).size().rename("n_type")
    keys = pd.DataFrame({
        "game_pk": events["game_pk"].to_numpy(), "at_bat_number": next_ab.to_numpy(),
        "pitcher": events["pitcher"].to_numpy(), "pitch_type": events["pitch_type"].to_numpy(),
    })
    keys = keys.join(n_all, on=["game_pk", "at_bat_number", "pitcher"]).join(
        n_type, on=["game_pk", "at_bat_number", "pitcher", "pitch_type"]
    )
    count = keys["next_ab_pitch_count"]
    share = keys["n_type"].fillna(0) / count
    return pd.DataFrame({
        "next_ab_pitch_count": count.fillna(0).astype(int).to_numpy(),
        "same_type_share": share.to_numpy(),
        "reused_same_type": (keys["n_type"].fillna(0) > 0).astype(float).where(count.notna()).to_numpy(),
    }, index=events.index)


def build_batted_ball_events(pitches: pd.DataFrame, season_usage: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Batted-ball events that have a comparison PA in `mode`, with xwOBA, barrel and avoidance columns.

    `pitches` must already carry `baseline_usage` (add_prior_game_usage) and the situation columns.
    """
    events = pitches[pitches["events"].isin(OUTCOME_GROUPS) & pitches["pitch_type"].notna()].copy()
    events = ELIGIBILITY[mode](pitches, events)
    next_ab = comparison_at_bat(pitches, events, mode)
    events = events.join(comparison_share(pitches, events, next_ab))
    events = events.merge(
        season_usage[["pitcher", "season", "pitch_type", "season_usage_rate"]],
        on=["pitcher", "season", "pitch_type"], how="left",
    )
    events["outcome"] = events["events"].map(OUTCOME_GROUPS)
    events["pitch_family"] = events["pitch_type"].map(map_pitch_family)
    events["barrel"] = is_barrel(events["launch_speed"], events["launch_angle"])
    return events


def add_xwoba_quartile(events: pd.DataFrame) -> pd.DataFrame:
    """Quartiles of xwOBA over all batted balls of the events (Q1 = weakest contact)."""
    out = events.copy()
    out["xwoba_q"] = pd.qcut(out[XWOBA], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return out


def cell_table(
    events: pd.DataFrame, baseline_col: str, by: str = "xwoba_q", ref: tuple = ("아웃", "Q1"), z: float = 1.96
) -> pd.DataFrame:
    """Mean diff per outcome x `by` cell and its difference from the reference cell
    (default: weak out = 아웃 in xwOBA Q1), with a normal-approximation 95% CI."""
    d = events[events[baseline_col].notna() & events[by].notna()].copy()
    d["diff"] = d[baseline_col] - d["same_type_share"]
    g = d.groupby(["outcome", by], observed=True)["diff"].agg(["size", "mean", "var"])
    r = g.loc[ref]
    g["vs_ref"] = g["mean"] - r["mean"]
    se = np.sqrt(g["var"] / g["size"] + r["var"] / r["size"])
    g["ci_low"], g["ci_high"] = g["vs_ref"] - z * se, g["vs_ref"] + z * se
    g = g.drop(columns="var").rename(columns={"size": "n", "mean": "diff_mean"}).reset_index()
    g["outcome"] = pd.Categorical(g["outcome"], OUTCOME_ORDER, ordered=True)
    return g.sort_values(["outcome", by]).reset_index(drop=True)
