import math

import pandas as pd
import pytest

from src.preprocessing.build_next_ab_dataset import (
    build_event_dataset,
    build_event_dataset_for_events,
    compute_baseline_usage,
    compute_catcher_changed,
    compute_platoon_match,
    compute_score_diff,
    filter_to_same_batter_rematch,
    find_next_at_bat_pitches,
    identify_events_by_outcome,
    identify_extra_base_hit_events,
)


def _pitch(**overrides):
    row = {
        "game_pk": 1, "game_date": "2024-06-01", "pitcher": 100,
        "player_name": "Test Pitcher", "batter": 200, "pitch_type": "FF",
        "events": None, "at_bat_number": 1, "pitch_number": 1,
        "balls": 0, "strikes": 0, "outs_when_up": 0, "inning": 1,
        "inning_topbot": "Top", "stand": "R", "p_throws": "R",
        "home_score": 0, "away_score": 0,
    }
    row.update(overrides)
    return row


def test_identify_extra_base_hit_events_filters_by_events_column():
    pitches = pd.DataFrame([
        _pitch(events="single"),
        _pitch(events="double"),
        _pitch(events="triple"),
        _pitch(events="home_run"),
        _pitch(events=None),
    ])
    result = identify_extra_base_hit_events(pitches)
    assert sorted(result["events"].tolist()) == ["double", "home_run", "triple"]


def test_find_next_at_bat_pitches_matches_only_target_at_bat_number():
    game_pitches = pd.DataFrame([
        _pitch(at_bat_number=3, pitch_number=1),
        _pitch(at_bat_number=4, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=4, pitch_number=2, pitch_type="FF"),
        _pitch(at_bat_number=5, pitch_number=1),
    ])
    next_ab = find_next_at_bat_pitches(game_pitches, next_at_bat_number=4)
    assert len(next_ab) == 2
    assert set(next_ab["pitch_type"]) == {"SL", "FF"}


def test_compute_score_diff_uses_pitching_team_perspective():
    top_row = pd.Series(_pitch(inning_topbot="Top", home_score=5, away_score=2))
    bot_row = pd.Series(_pitch(inning_topbot="Bot", home_score=5, away_score=2))
    assert compute_score_diff(top_row) == 3  # home team pitching: 5 - 2
    assert compute_score_diff(bot_row) == -3  # away team pitching: 2 - 5


def test_compute_baseline_usage_is_nan_when_no_prior_pitches():
    game_pitches = pd.DataFrame([_pitch(at_bat_number=1, pitch_number=1, pitch_type="FF")])
    usage = compute_baseline_usage(
        game_pitches, before_at_bat_number=1, before_pitch_number=1, hit_pitch_type="FF"
    )
    assert math.isnan(usage)


def test_compute_baseline_usage_counts_earlier_at_bats_and_earlier_pitches_in_same_at_bat():
    game_pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="FF"),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="SL"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="FF"),  # earlier pitch, same AB as the hit
        _pitch(at_bat_number=2, pitch_number=2, pitch_type="FF"),  # the hit pitch itself: excluded
    ])
    usage = compute_baseline_usage(
        game_pitches, before_at_bat_number=2, before_pitch_number=2, hit_pitch_type="FF"
    )
    assert usage == 2 / 3  # FF at (1,1) and (2,1) out of 3 prior pitches


def test_build_event_dataset_marks_reuse_when_next_ab_repeats_pitch_type():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="FF", events="home_run"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=2, pitch_number=2, pitch_type="FF"),
        _pitch(at_bat_number=2, pitch_number=3, pitch_type="FF"),
    ])
    result = build_event_dataset(pitches)
    assert len(result) == 1
    event = result.iloc[0]
    assert event["has_next_ab"] == True
    assert event["next_ab_pitch_count"] == 3
    assert event["reused_same_type"] == 1
    assert event["same_type_share"] == 2 / 3
    assert event["hit_pitch_type"] == "FF"
    assert event["pitch_family"] == "fastball"
    assert event["baseline_usage"] == 0.0  # only prior pitch was SL


def test_build_event_dataset_flags_missing_next_ab_without_dropping_row():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="FF", events="double"),
        # pitcher is pulled: at_bat_number 2 exists but is thrown by a different pitcher
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL", pitcher=999),
    ])
    result = build_event_dataset(pitches)
    assert len(result) == 1
    event = result.iloc[0]
    assert event["has_next_ab"] == False
    assert event["next_ab_pitch_count"] == 0
    assert math.isnan(event["reused_same_type"])
    assert math.isnan(event["same_type_share"])


def test_identify_events_by_outcome_filters_by_arbitrary_event_set():
    pitches = pd.DataFrame([
        _pitch(events="field_out"),
        _pitch(events="double"),
        _pitch(events="strikeout"),
        _pitch(events="field_out"),
    ])
    result = identify_events_by_outcome(pitches, {"field_out"})
    assert len(result) == 2
    assert set(result["events"]) == {"field_out"}


def test_build_event_dataset_for_events_computes_identical_logic_for_a_placebo_outcome():
    """A control-group outcome (e.g. field_out) must go through the exact
    same feature computation as an extra-base hit -- only which rows are
    selected as "events" differs.
    """
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="FF", events="field_out"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=2, pitch_number=2, pitch_type="FF"),
        _pitch(at_bat_number=2, pitch_number=3, pitch_type="FF"),
    ])
    placebo_events = identify_events_by_outcome(pitches, {"field_out"})
    result = build_event_dataset_for_events(pitches, placebo_events)

    assert len(result) == 1
    event = result.iloc[0]
    assert event["events"] == "field_out"
    assert event["has_next_ab"] == True
    assert event["next_ab_pitch_count"] == 3
    assert event["reused_same_type"] == 1
    assert event["same_type_share"] == 2 / 3
    assert event["hit_pitch_type"] == "FF"
    assert event["baseline_usage"] == 0.0  # only prior pitch was SL

    # build_event_dataset() (the XBH-only wrapper) must still behave exactly as before
    xbh_only = build_event_dataset(pitches)
    assert len(xbh_only) == 0  # no double/triple/home_run in this fixture


def test_compute_catcher_changed_detects_change():
    assert compute_catcher_changed(111, 222) == 1


def test_compute_catcher_changed_detects_no_change():
    assert compute_catcher_changed(111, 111) == 0


def test_compute_catcher_changed_is_nan_when_either_value_missing():
    assert math.isnan(compute_catcher_changed(float("nan"), 111))
    assert math.isnan(compute_catcher_changed(111, float("nan")))


def test_build_event_dataset_omits_catcher_changed_when_fielder_2_absent():
    """Backward compatibility: pitches without a fielder_2 column (e.g. all
    existing fixtures/raw data collected before it was added) must keep
    working exactly as before, with no catcher_changed column at all.
    """
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="FF", events="home_run"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="FF"),
    ])
    result = build_event_dataset(pitches)
    assert "catcher_changed" not in result.columns


def test_build_event_dataset_adds_catcher_changed_when_fielder_2_present():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="SL", fielder_2=500),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="FF", events="home_run", fielder_2=500),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="FF", fielder_2=777),
        _pitch(at_bat_number=2, pitch_number=2, pitch_type="SL", fielder_2=777),
    ])
    result = build_event_dataset(pitches)
    assert len(result) == 1
    assert result.iloc[0]["catcher_changed"] == 1  # 500 (hit pitch) -> 777 (next AB's first pitch)


def test_build_event_dataset_catcher_changed_is_nan_without_next_ab():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="FF", events="double", fielder_2=500),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL", pitcher=999, fielder_2=500),
    ])
    result = build_event_dataset(pitches)
    assert len(result) == 1
    assert math.isnan(result.iloc[0]["catcher_changed"])


def test_reuse_metrics_are_nan_when_the_event_pitch_has_no_pitch_type():
    """A pitch with no pitch_type has no "same type" to compare: it must not
    count as 'not reused' (reused=0), 0% baseline usage, or match a next-AB
    pitch that is also missing (NaN == NaN is False, but 0.0 would leak in).
    """
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type=None, events="double"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=2, pitch_number=2, pitch_type=None),
    ])
    event = build_event_dataset(pitches).iloc[0]
    assert event["has_next_ab"] == True
    assert event["next_ab_pitch_count"] == 2
    assert math.isnan(event["reused_same_type"])
    assert math.isnan(event["same_type_share"])
    assert math.isnan(event["baseline_usage"])


def test_compute_platoon_match_is_1_for_same_handedness_and_0_for_opposite():
    assert compute_platoon_match("R", "R") == 1
    assert compute_platoon_match("L", "L") == 1
    assert compute_platoon_match("L", "R") == 0
    assert compute_platoon_match("R", "L") == 0
    assert math.isnan(compute_platoon_match(None, "R"))
    assert math.isnan(compute_platoon_match("R", float("nan")))


def test_event_record_carries_platoon_match_event_pitch_number_and_compared_at_bat():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="SL", stand="L", p_throws="R"),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="FF", stand="L", p_throws="R", events="home_run"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL", batter=300),
        _pitch(at_bat_number=10, pitch_number=1, pitch_type="FF", stand="L", p_throws="R"),
    ])
    events = identify_extra_base_hit_events(pitches)

    default = build_event_dataset_for_events(pitches, events).iloc[0]
    assert default["platoon_match"] == 0  # left-handed batter vs right-handed pitcher
    assert default["event_pitch_number"] == 2
    assert default["next_at_bat_number"] == 2

    rematch = build_event_dataset_for_events(pitches, events, next_pa_mode="same_batter").iloc[0]
    assert rematch["next_at_bat_number"] == 10

    lone = pitches.iloc[:2]  # no later PA at all
    none_row = build_event_dataset_for_events(lone, identify_extra_base_hit_events(lone)).iloc[0]
    assert math.isnan(none_row["next_at_bat_number"])


def _rematch_fixture():
    return pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="FF", batter=200, events="home_run"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL", batter=300),  # next batter, not a rematch
        _pitch(at_bat_number=10, pitch_number=1, pitch_type="FF", batter=200),  # batter 200 returns
        _pitch(at_bat_number=10, pitch_number=2, pitch_type="SL", batter=200),
    ])


def test_same_batter_mode_uses_the_batters_next_pa_not_the_next_batter():
    pitches = _rematch_fixture()
    events = identify_extra_base_hit_events(pitches)

    rematch = build_event_dataset_for_events(pitches, events, next_pa_mode="same_batter").iloc[0]
    assert rematch["has_next_ab"] == True
    assert rematch["next_ab_pitch_count"] == 2  # AB 10, not AB 2
    assert rematch["same_type_share"] == 0.5
    assert rematch["reused_same_type"] == 1

    default = build_event_dataset_for_events(pitches, events).iloc[0]  # unchanged default behavior
    assert default["next_ab_pitch_count"] == 1  # AB 2 (batter 300)
    assert default["reused_same_type"] == 0


def test_same_batter_mode_flags_missing_when_batter_faces_a_different_pitcher_next():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="FF", batter=200, events="double"),
        _pitch(at_bat_number=10, pitch_number=1, pitch_type="FF", batter=200, pitcher=999),
    ])
    result = build_event_dataset_for_events(
        pitches, identify_extra_base_hit_events(pitches), next_pa_mode="same_batter"
    )
    assert result.iloc[0]["has_next_ab"] == False
    assert result.iloc[0]["next_ab_pitch_count"] == 0
    assert math.isnan(result.iloc[0]["reused_same_type"])


def test_same_batter_mode_flags_missing_when_batter_never_returns():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="FF", batter=200, events="double"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL", batter=300),
    ])
    result = build_event_dataset_for_events(
        pitches, identify_extra_base_hit_events(pitches), next_pa_mode="same_batter"
    )
    assert result.iloc[0]["has_next_ab"] == False


def test_unknown_next_pa_mode_is_rejected():
    pitches = _rematch_fixture()
    with pytest.raises(ValueError):
        build_event_dataset_for_events(pitches, identify_extra_base_hit_events(pitches), next_pa_mode="bogus")


def test_filter_to_same_batter_rematch_keeps_only_events_with_a_same_pitcher_rematch():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, batter=200, events="field_out"),  # returns at AB 5 vs same pitcher -> keep
        _pitch(at_bat_number=2, batter=300, events="field_out"),  # never returns -> drop
        _pitch(at_bat_number=3, batter=400, events="field_out"),  # returns at AB 9 vs another pitcher -> drop
        _pitch(at_bat_number=5, batter=200),
        _pitch(at_bat_number=9, batter=400, pitcher=999),
    ])
    candidates = identify_events_by_outcome(pitches, {"field_out"})
    kept = filter_to_same_batter_rematch(pitches, candidates)
    assert kept["at_bat_number"].tolist() == [1]
    assert set(kept.index) <= set(candidates.index)
