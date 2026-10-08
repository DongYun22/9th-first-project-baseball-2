import numpy as np
import pandas as pd

from src.analysis.batter_specific import (
    event_summary,
    hand_season_usage,
    pa_avoidance,
    pa_table,
    prior_same_batter_avoidance,
)


def _pitches():
    # pitcher 10, one game. Batter 1 (R) hits an XBH on SL in PA 1, then batters 2 (L) and 3 (R)
    # come up, then batter 1 again in PA 4 (rematch). PA 5 is after the rematch and must be ignored.
    rows = [
        (1, 1, 1, "R", "SL"), (1, 2, 1, "R", "SL"),
        (2, 1, 2, "L", "SL"), (2, 2, 2, "L", "FF"),
        (3, 1, 3, "R", "FF"), (3, 2, 3, "R", "FF"), (3, 3, 3, "R", None), (3, 4, 3, "R", "SL"),
        (4, 1, 1, "R", "FF"), (4, 2, 1, "R", "FF"),
        (5, 1, 4, "R", "SL"),
    ]
    df = pd.DataFrame(rows, columns=["at_bat_number", "pitch_number", "batter", "stand", "pitch_type"])
    df["game_pk"], df["pitcher"], df["season"] = 1, 10, 2025
    return df


def test_hand_season_usage_counts_missing_types_in_denominator():
    u = hand_season_usage(_pitches()).set_index(["stand", "pitch_type"])["hand_rate"]
    assert np.isclose(u[("R", "SL")], 4 / 9)  # 9 pitches to R batters, one has no type
    assert np.isclose(u[("L", "SL")], 1 / 2)


def test_pa_avoidance_and_event_summary():
    p = _pitches()
    pas, by_type = pa_table(p)
    hand = pd.DataFrame({"pitcher": 10, "season": 2025, "stand": ["R", "L"], "pitch_type": "SL",
                         "hand_rate": [0.5, 0.25]})
    events = pd.DataFrame({"event_id": [0], "game_pk": [1], "pitcher": [10], "at_bat_number": [1],
                           "next_at_bat_number": [4.0], "hit_pitch_type": ["SL"]})
    pa = pa_avoidance(events, pas, by_type, hand)
    assert pa["at_bat_number"].tolist() == [2, 3, 4]
    assert pa["role"].tolist() == ["intervening", "intervening", "rematch"]
    assert pa["k"].tolist() == [1, 2, 3]
    # PA 2 (L): 0.25 - 1/2 ; PA 3 (R): 0.5 - 1/4 ; rematch (R): 0.5 - 0
    assert np.allclose(pa["avoid"], [-0.25, 0.25, 0.5])

    s = event_summary(pa).iloc[0]
    # intervening pooled: expected 0.25*2 + 0.5*4 = 2.5, actual SL = 2, over 6 pitches
    assert s["n_inter_pitches"] == 6
    assert np.isclose(s["avoid_inter"], (2.5 - 2) / 6)
    assert np.isclose(s["avoid_rematch"], 0.5)
    assert np.isclose(s["batter_specific"], 0.5 - 0.5 / 6)


def test_prior_same_batter_avoidance_uses_only_earlier_pas_of_that_batter():
    p = _pitches()
    pas, by_type = pa_table(p)
    hand = pd.DataFrame({"pitcher": 10, "season": 2025, "stand": "R", "pitch_type": "FF", "hand_rate": [0.5]})
    # event = batter 1's PA 4 (T = FF); his only earlier PA is PA 1 (2 pitches, no FF)
    events = pd.DataFrame({"event_id": [0, 1], "game_pk": 1, "pitcher": 10, "batter": [1, 2],
                           "at_bat_number": [4, 2], "hit_pitch_type": "FF"})
    r = prior_same_batter_avoidance(events, pas, by_type, hand)
    assert r["event_id"].tolist() == [0]  # batter 2 had no earlier PA
    assert r.loc[0, "n_prior_pitches"] == 2
    assert np.isclose(r.loc[0, "avoid_prior"], 0.5)
