"""Is the avoidance aimed at the batter who hit it, or at the pitch in general?

Within one event (XBH or matched control), compare how much the pitcher avoids the event
pitch type T against the batters faced IN BETWEEN (after the event, before the rematch) with
how much he avoids it in the REMATCH PA against the batter who produced the event.

Avoidance of a PA = expected share - actual share of T, where expected = the pitcher's season
usage of T against batters of that PA's handedness, so platoon differences between the
intervening batters and the hitter do not show up as avoidance.

batter_specific = avoid_rematch - avoid_intervening (> 0: avoids T more against that batter).
Comparing it between XBH and control removes what the next time through the order does anyway.
"""

from __future__ import annotations

import pandas as pd


def hand_season_usage(pitches: pd.DataFrame) -> pd.DataFrame:
    """Share of each pitch type among a pitcher's season pitches to batters of each hand
    (pitches with a missing type count in the denominator, as in the comparison-PA share)."""
    total = pitches.groupby(["pitcher", "season", "stand"]).size().rename("n")
    by_type = pitches.groupby(["pitcher", "season", "stand", "pitch_type"]).size().rename("n_type")
    out = by_type.reset_index().join(total, on=["pitcher", "season", "stand"])
    out["hand_rate"] = out["n_type"] / out["n"]
    return out[["pitcher", "season", "stand", "pitch_type", "hand_rate"]]


def pa_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, PA, pitcher, pitch type) with the PA's pitch count and batter hand."""
    keys = ["game_pk", "at_bat_number", "pitcher"]
    pa = pitches.groupby(keys).agg(n_pitches=("pitch_number", "size"), stand=("stand", "first"),
                                   season=("season", "first"), batter=("batter", "first"))
    by_type = pitches.groupby(keys + ["pitch_type"]).size().rename("n_type").reset_index()
    return pa.reset_index(), by_type


def pa_avoidance(
    events: pd.DataFrame, pas: pd.DataFrame, by_type: pd.DataFrame, hand_usage: pd.DataFrame
) -> pd.DataFrame:
    """Every PA of the event's pitcher strictly between the event PA and the rematch PA
    (role='intervening', with its order k = 1, 2, ...) plus the rematch PA itself (role='rematch'),
    each with its expected and actual share of the event pitch type T."""
    ev = events[["event_id", "game_pk", "pitcher", "at_bat_number", "next_at_bat_number", "hit_pitch_type"]].rename(
        columns={"at_bat_number": "event_ab", "next_at_bat_number": "rematch_ab", "hit_pitch_type": "pitch_type"}
    )
    m = ev.merge(pas, on=["game_pk", "pitcher"])
    m = m[(m["at_bat_number"] > m["event_ab"]) & (m["at_bat_number"] <= m["rematch_ab"])].copy()
    m["role"] = (m["at_bat_number"] == m["rematch_ab"]).map({True: "rematch", False: "intervening"})
    m = m.sort_values(["event_id", "at_bat_number"])
    m["k"] = m.groupby("event_id").cumcount() + 1
    m = m.merge(by_type, on=["game_pk", "at_bat_number", "pitcher", "pitch_type"], how="left")
    m = m.merge(hand_usage, on=["pitcher", "season", "stand", "pitch_type"], how="left")
    m["n_type"] = m["n_type"].fillna(0)
    m["hand_rate"] = m["hand_rate"].fillna(0)
    m["share"] = m["n_type"] / m["n_pitches"]
    m["avoid"] = m["hand_rate"] - m["share"]
    return m


def event_summary(pa: pd.DataFrame) -> pd.DataFrame:
    """Per event: pitch-weighted avoidance over the intervening PAs vs the rematch PA."""
    pa = pa.assign(exp_pitches=pa["hand_rate"] * pa["n_pitches"])
    g = pa.groupby(["event_id", "role"])[["n_pitches", "n_type", "exp_pitches"]].sum().unstack("role")
    out = pd.DataFrame({
        "n_inter_pitches": g[("n_pitches", "intervening")],
        "avoid_inter": (g[("exp_pitches", "intervening")] - g[("n_type", "intervening")]) / g[("n_pitches", "intervening")],
        "avoid_rematch": (g[("exp_pitches", "rematch")] - g[("n_type", "rematch")]) / g[("n_pitches", "rematch")],
    })
    out["batter_specific"] = out["avoid_rematch"] - out["avoid_inter"]
    return out.dropna().reset_index()


def prior_same_batter_avoidance(
    events: pd.DataFrame, pas: pd.DataFrame, by_type: pd.DataFrame, hand_usage: pd.DataFrame
) -> pd.DataFrame:
    """Pitch-weighted avoidance of T against the SAME batter in this pitcher's PAs BEFORE the
    event PA (same game). Events with no earlier PA against that batter are dropped. If the
    pitcher was already avoiding T against this batter, the rematch avoidance is not a reaction."""
    ev = events[["event_id", "game_pk", "pitcher", "batter", "at_bat_number", "hit_pitch_type"]].rename(
        columns={"at_bat_number": "event_ab", "hit_pitch_type": "pitch_type"}
    )
    m = ev.merge(pas, on=["game_pk", "pitcher", "batter"])
    m = m[m["at_bat_number"] < m["event_ab"]]
    m = m.merge(by_type, on=["game_pk", "at_bat_number", "pitcher", "pitch_type"], how="left")
    m = m.merge(hand_usage, on=["pitcher", "season", "stand", "pitch_type"], how="left")
    m["n_type"] = m["n_type"].fillna(0)
    m["exp_pitches"] = m["hand_rate"].fillna(0) * m["n_pitches"]
    g = m.groupby("event_id")[["n_pitches", "n_type", "exp_pitches"]].sum()
    return pd.DataFrame({
        "n_prior_pitches": g["n_pitches"],
        "avoid_prior": (g["exp_pitches"] - g["n_type"]) / g["n_pitches"],
    }).reset_index()
