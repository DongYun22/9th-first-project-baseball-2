import numpy as np
import pandas as pd
import pytest

from src.analysis.hit_quality import (
    add_prior_game_usage,
    add_xwoba_quartile,
    build_batted_ball_events,
    cell_table,
    is_barrel,
)
from src.preprocessing.build_next_ab_dataset import build_event_dataset_for_events


def _pitches():
    # one game, pitcher 10 faces batters 1,2,1 (rematch) then pitcher 20 takes over
    rows = [
        # ab, pitch, pitcher, batter, type, events
        (1, 1, 10, 1, "FF", None), (1, 2, 10, 1, "SL", "double"),
        (2, 1, 10, 2, "SL", None), (2, 2, 10, 2, "FF", None), (2, 3, 10, 2, None, None), (2, 4, 10, 2, "FF", "field_out"),
        (3, 1, 10, 1, "SL", None), (3, 2, 10, 1, "CH", "single"),
        (4, 1, 20, 3, "FF", "home_run"),
        (5, 1, 20, 2, "FF", "field_out"),
    ]
    df = pd.DataFrame(rows, columns=["at_bat_number", "pitch_number", "pitcher", "batter", "pitch_type", "events"])
    df["game_pk"], df["season"], df["game_date"] = 1, 2025, "2025-05-01"
    df["player_name"], df["stand"], df["p_throws"] = "x", "R", "R"
    for c in ["balls", "strikes", "outs_when_up", "inning", "home_score", "away_score"]:
        df[c] = 0
    df["inning_topbot"] = "Top"
    df["launch_speed"], df["launch_angle"], df["estimated_woba_using_speedangle"] = 100.0, 25.0, 0.5
    return df


@pytest.mark.parametrize("mode", ["same_batter", "next_batter"])
def test_vectorized_share_matches_event_builder(mode):
    pitches = add_prior_game_usage(_pitches())
    season_usage = pd.DataFrame({"pitcher": [10, 20], "season": 2025, "pitch_type": ["SL", "FF"],
                                 "season_usage_rate": [0.4, 1.0]})
    fast = build_batted_ball_events(pitches, season_usage, mode).set_index("at_bat_number")
    slow = build_event_dataset_for_events(
        pitches, pitches[pitches["at_bat_number"].isin(fast.index) & pitches["events"].notna()], next_pa_mode=mode
    ).set_index("at_bat_number")
    for col in ["same_type_share", "reused_same_type", "next_ab_pitch_count", "baseline_usage"]:
        np.testing.assert_allclose(fast[col].to_numpy(float), slow.loc[fast.index, col].to_numpy(float), err_msg=col)


def test_eligibility_per_mode():
    pitches = add_prior_game_usage(_pitches())
    su = pd.DataFrame(columns=["pitcher", "season", "pitch_type", "season_usage_rate"])
    # rematch: only the double (ab 1 -> batter 1 again in ab 3, same pitcher)
    assert build_batted_ball_events(pitches, su, "same_batter")["at_bat_number"].tolist() == [1]
    # next batter: ab 1 -> 2 and 2 -> 3 (pitcher 10), 4 -> 5 (pitcher 20); ab 3 -> 4 is a new pitcher
    assert sorted(build_batted_ball_events(pitches, su, "next_batter")["at_bat_number"]) == [1, 2, 4]


def test_is_barrel():
    ev = pd.Series([98.0, 98.0, 110.0, 95.0, np.nan])
    la = pd.Series([26.0, 40.0, 20.0, 26.0, 20.0])
    r = is_barrel(ev, la)
    assert r.iloc[:4].tolist() == [True, False, True, False]
    assert pd.isna(r.iloc[4])


def test_cell_table_reference_is_weak_out():
    n = 40
    ev = pd.DataFrame({
        "outcome": ["아웃"] * (2 * n) + ["홈런"] * n,
        "estimated_woba_using_speedangle": list(np.linspace(0, 0.2, n)) + list(np.linspace(0.8, 1, n)) * 2,
        "b": 0.5,
        "same_type_share": [0.5] * n + [0.4] * n + [0.2] * n,
    })
    t = cell_table(add_xwoba_quartile(ev), "b").set_index(["outcome", "xwoba_q"])
    assert t.loc[("아웃", "Q1"), "vs_ref"] == 0
    assert np.isclose(t.loc[("홈런", "Q4"), "vs_ref"], 0.3)
