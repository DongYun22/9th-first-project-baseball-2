import numpy as np
import pandas as pd

from src.analysis.usage_change_factors import build_design_matrix, build_same_season_rematch_events


def _pitches():
    # pitcher 10 vs batter 7: a home run on an SL (2021 game 1), a rematch in game 2 (SL, FF), then 2022
    rows = [
        (2021, "2021-04-01", 1, 3, 1, "SL", "home_run"),
        (2021, "2021-04-08", 2, 5, 1, "SL", None),
        (2021, "2021-04-08", 2, 5, 2, "FF", "field_out"),
        (2022, "2022-04-01", 3, 2, 1, "SL", "double"),
    ]
    df = pd.DataFrame(rows, columns=["season", "game_date", "game_pk", "at_bat_number", "pitch_number", "pitch_type", "events"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.assign(pitcher=10, batter=7)


def _usage():
    return pd.DataFrame({"pitcher": [10, 10], "season": [2021, 2022], "pitch_type": ["SL", "SL"],
                         "season_usage_rate": [0.3, 0.4]})


def test_rematch_is_the_next_pa_against_the_same_batter_within_the_season():
    events = build_same_season_rematch_events(_pitches(), _usage())
    assert len(events) == 1  # the 2022 double has no later PA that season
    row = events.iloc[0]
    assert row["game_pk"] == 1 and row["next_game_pk"] == 2 and row["next_at_bat_number"] == 5
    assert np.isclose(row["post_usage"], 0.5) and np.isclose(row["usage_change"], 0.5 - 0.3)


def test_an_event_whose_rematch_has_no_typed_pitch_is_left_out():
    pitches = _pitches()
    pitches.loc[pitches["game_pk"] == 2, "pitch_type"] = None
    assert build_same_season_rematch_events(pitches, _usage()).empty


def _events():
    return pd.DataFrame({
        "events": ["double", "home_run", "triple"], "launch_speed": [100.0, 105.0, 98.0], "launch_angle": [15.0, 28.0, 20.0],
        "estimated_woba_using_speedangle": [0.6, 1.9, 0.7], "pitch_type": ["FF", "SL", "KN"],
        "release_speed": [95.0, 85.0, 70.0], "pfx_x": [-0.5, 0.3, 0.1], "pfx_z": [1.4, 0.2, 0.0],
        "plate_x": [0.0, 0.1, 0.2], "plate_z": [2.5, 2.0, 2.2], "p_throws": ["R", "L", "R"], "stand": ["L", "L", "R"],
        "balls": [0, 3, 1], "strikes": [0, 2, 1], "outs_when_up": [0, 1, 2],
        "on_1b": [np.nan, 5.0, np.nan], "on_2b": [np.nan, np.nan, np.nan], "on_3b": [np.nan, np.nan, 9.0],
        "score_diff": [2, -1, 0], "usage_change": [-0.1, -0.3, 0.05],
    })


def test_design_matrix_orients_movement_and_score_and_one_hot_encodes():
    X, y = build_design_matrix(_events())
    assert X["pfx_x_arm"].tolist() == [-0.5, -0.3, 0.1]          # a lefty's horizontal movement is mirrored
    assert X["score_diff_bat"].tolist() == [-2, 1, 0]            # batting team's view
    assert X["runner_1b"].tolist() == [0, 1, 0] and X["runner_3b"].tolist() == [0, 0, 1]
    assert X["events_home_run"].tolist() == [False, True, False] and "events_double" not in X  # first level dropped
    assert X["matchup_R-L"].tolist() == [True, False, False]
    assert X["count_state_3-2"].tolist() == [False, True, False]
    assert X["pitch_group_Other"].tolist() == [False, False, True]  # a rare type is pooled
    assert y.tolist() == [-0.1, -0.3, 0.05]
    assert not X.select_dtypes(include=["object", "category"]).shape[1]


def test_design_matrix_drops_rows_with_a_missing_input():
    events = _events()
    events.loc[0, "launch_speed"] = np.nan
    X, y = build_design_matrix(events)
    assert len(X) == 2 and len(y) == 2
