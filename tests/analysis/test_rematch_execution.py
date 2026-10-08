import numpy as np
import pandas as pd

from src.analysis.rematch_execution import (
    BASE_KEYS,
    MEASURES,
    deviation_from_typical,
    group_difference_table,
    mean_ci,
    typical_tables,
)


def _pitches():
    # 30 pitches in other games at 0.0, the event PA (0.2 then the event pitch 0.0), the rematch PA (0.5 twice)
    rows = [{"game_pk": 100 + i, "at_bat_number": 1, "pitch_number": 1, "v": 0.0} for i in range(30)]
    rows += [
        {"game_pk": 1, "at_bat_number": 5, "pitch_number": 1, "v": 0.2},
        {"game_pk": 1, "at_bat_number": 5, "pitch_number": 2, "v": 0.0},
        {"game_pk": 1, "at_bat_number": 9, "pitch_number": 1, "v": 0.5},
        {"game_pk": 1, "at_bat_number": 9, "pitch_number": 2, "v": 0.5},
    ]
    df = pd.DataFrame(rows).assign(pitcher=10, season=2021, pitch_type="SL", stand="R")
    for m in MEASURES:
        df[m] = df["v"]
    return df


def _events():
    return pd.DataFrame({
        "game_pk": [1], "pitcher": [10], "season": [2021], "hit_pitch_type": ["SL"], "p_throws": ["R"],
        "at_bat_number": [5], "event_pitch_number": [2], "next_at_bat_number": [9.0],
    })


def test_rematch_deviation_leaves_the_rematch_pa_out_of_the_pitchers_norm():
    pitches = _pitches()
    base_sum, base_cnt = typical_tables(pitches)
    dev = deviation_from_typical(_events(), pitches, base_sum, base_cnt, "rematch")
    assert np.isclose(dev.loc[0, "plate_x"], 0.5 - 0.2 / 32)
    assert dev.loc[0, "hit_pitch_type"] == "SL" and dev.loc[0, "p_throws"] == "R"


def test_event_deviation_drops_the_pitch_that_became_the_event():
    pitches = _pitches()
    base_sum, base_cnt = typical_tables(pitches)
    dev = deviation_from_typical(_events(), pitches, base_sum, base_cnt, "event")
    assert np.isclose(dev.loc[0, "plate_x"], 0.2 - 1.0 / 33)


def test_deviation_is_missing_when_fewer_than_30_other_pitches_define_the_norm():
    pitches = _pitches()
    pitches = pitches[pitches["game_pk"] < 110]
    grouped = pitches.groupby(BASE_KEYS)[MEASURES]
    dev = deviation_from_typical(_events(), pitches, grouped.sum(), grouped.count(), "rematch")
    assert dev["plate_x"].isna().all()


def test_mean_ci_is_the_normal_approximation_interval():
    m, lo, hi = mean_ci(pd.Series([0.0, 2.0, np.nan]))
    half = 1.96 * np.sqrt(2.0) / np.sqrt(2)
    assert np.isclose(m, 1.0) and np.isclose(lo, 1.0 - half) and np.isclose(hi, 1.0 + half)


def test_group_difference_table_scales_movement_to_inches_and_skips_small_cells():
    dx = pd.DataFrame({"p_throws": ["R"] * 3, "hit_pitch_type": ["SL"] * 3, **{c: [0.3, 0.5, 0.4] for c in MEASURES}})
    dc = pd.DataFrame({"p_throws": ["R"] * 3, "hit_pitch_type": ["SL"] * 3, **{c: [0.1, 0.3, 0.2] for c in MEASURES}})
    table = group_difference_table(dx, dc, min_events=3)
    row = table[table["measure"] == "plate_x"].iloc[0]
    assert np.isclose(row["xbh"], 0.4) and np.isclose(row["control"], 0.2) and np.isclose(row["diff"], 0.2)
    assert np.isclose(table[table["measure"] == "pfx_x"].iloc[0]["diff"], 0.2 * 12)
    assert bool(row["distinct"]) == (row["ci_low"] > 0 or row["ci_high"] < 0)
    assert group_difference_table(dx, dc, min_events=4).empty
