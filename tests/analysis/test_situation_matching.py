import numpy as np
import pandas as pd

from src.analysis.situation_matching import (
    add_situation_columns,
    attach_event_situation,
    attach_next_situation,
    balance_table,
    filter_to_next_batter,
    pure_reduction,
    runner_group,
    smd,
    stratified_reduction,
)


def test_add_situation_columns():
    df = pd.DataFrame({
        "on_1b": [np.nan, 1.0, np.nan, 5.0], "on_2b": [np.nan, np.nan, 2.0, 6.0], "on_3b": [np.nan, np.nan, np.nan, 7.0],
        "outs_when_up": [0, 1, 2, 2], "inning": [1, 4, 9, 7], "balls": [0, 2, 1, 3], "strikes": [0, 1, 2, 1],
        "inning_topbot": ["Top", "Bot", "Top", "Bot"], "home_score": [5, 5, 4, 1], "away_score": [2, 2, 4, 6],
    })
    r = add_situation_columns(df)
    assert r["runners_n"].tolist() == [0, 1, 1, 3]
    assert r["risp"].tolist() == [0, 0, 1, 1]
    assert r["base_out_state"].tolist() == [0, 4, 8, 23]
    assert r["inning_bucket"].tolist() == ["1-3", "4-6", "7+", "7+"]
    assert r["count_bucket"].tolist() == ["even", "batter", "pitcher", "batter"]
    assert r["score_diff"].tolist() == [3, -3, 0, 5]
    assert r["score_diff_bucket"].tolist() == ["앞섬", "뒤짐", "동점", "앞섬"]


def _situation_pitches():
    return pd.DataFrame({
        "game_pk": [1] * 5, "pitcher": [10] * 5, "at_bat_number": [5, 5, 9, 9, 14], "pitch_number": [1, 3, 1, 2, 4],
        "runners_n": [0, 2, 1, 1, 3], "risp": [0, 1, 0, 0, 1], "bases": [0, 5, 1, 1, 7],
        "base_out_state": [0, 15, 3, 3, 23], "inning_bucket": ["1-3"] * 5, "count_bucket": ["even"] * 5,
        "score_diff": [1, -2, 0, 0, 3], "score_diff_bucket": ["앞섬", "뒤짐", "동점", "동점", "앞섬"],
        "outs_when_up": [0, 0, 1, 1, 2],
    })


def test_attach_event_and_next_situation():
    ev = pd.DataFrame({
        "game_pk": [1, 1], "pitcher": [10, 10], "at_bat_number": [5, 9],
        "event_pitch_number": [3, 2], "next_at_bat_number": [9.0, 14.0],
    })
    px = _situation_pitches()
    a = attach_event_situation(ev, px)
    assert a["runners_n"].tolist() == [2, 1]
    assert a["score_diff_bucket"].tolist() == ["뒤짐", "동점"]
    b = attach_next_situation(a, px)
    assert b["runners_n_next"].tolist() == [1, 3]
    assert b["outs_next"].tolist() == [1, 2]
    assert b["score_diff_next"].tolist() == [0, 3]


def test_filter_to_next_batter_keeps_only_same_pitcher_next_pa():
    pitches = pd.DataFrame({
        "game_pk": [1, 1, 1, 1], "at_bat_number": [1, 2, 3, 4], "pitcher": [10, 10, 10, 20],
    })
    events = pd.DataFrame({"game_pk": [1, 1, 1], "at_bat_number": [1, 3, 4], "pitcher": [10, 10, 20]})
    # PA 1 -> PA 2 same pitcher (keep); PA 3 -> PA 4 different pitcher (drop); PA 4 -> none (drop)
    assert filter_to_next_batter(pitches, events)["at_bat_number"].tolist() == [1]


def test_smd_and_balance_table():
    assert np.isclose(smd(pd.Series([0.0, 2.0]), pd.Series([1.0, 3.0])), -1 / np.sqrt(2))
    assert smd(pd.Series([1.0, 1.0]), pd.Series([1.0, 1.0])) == 0.0
    bt = balance_table(pd.DataFrame({"a": [0.0, 2.0]}), pd.DataFrame({"a": [1.0, 3.0]}), ["a"])
    assert np.isclose(bt.loc["a", "smd"], -1 / np.sqrt(2))


def test_pure_reduction_drops_rows_without_comparison_pa():
    x = pd.DataFrame({"has_next_ab": [True, True, False], "b": [0.5, 0.5, 0.5], "same_type_share": [0.3, 0.1, 0.0]})
    p = pd.DataFrame({"has_next_ab": [True, True], "b": [0.5, 0.5], "same_type_share": [0.5, 0.3]})
    r = pure_reduction(x, p, "b")
    assert r["n_xbh"] == 2 and r["n_placebo"] == 2
    assert np.isclose(r["net"], 0.2)


def test_runner_group_levels_do_not_overlap():
    g = runner_group(pd.DataFrame({"risp": [0, 0, 1, 1], "runners_n": [0, 1, 1, 2]}))
    assert g.tolist() == ["주자 없음", "1루만", "득점권", "득점권"]


def test_stratified_reduction_skips_small_cells():
    x = pd.DataFrame({"has_next_ab": True, "b": 0.5, "same_type_share": [0.1] * 40 + [0.2] * 5,
                      "g": ["A"] * 40 + ["B"] * 5})
    p = pd.DataFrame({"has_next_ab": True, "b": 0.5, "same_type_share": [0.3] * 40 + [0.4] * 40,
                      "g": ["A"] * 40 + ["B"] * 40})
    r = stratified_reduction(x, p, "b", "g", ["A", "B"])
    assert r["g"].tolist() == ["A"]
    assert np.isclose(r.loc[0, "net"], 0.2)
