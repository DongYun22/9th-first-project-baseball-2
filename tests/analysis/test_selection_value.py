import numpy as np
import pandas as pd

from src.analysis.selection_value import (
    FEATURES_SEASON,
    add_platoon_match,
    build_decisive_table,
    build_pa_pitches,
    build_rematch_events,
    group_gap,
    permutation_test_gap,
    refit_gap_draws,
    score_pa,
)


class _Stub:
    def predict(self, X):
        return np.where(X["pitch_type"].astype(str) == "SL", 0.5, 0.2)


def _pitches():
    # game 1, pitcher 10: PA 1 ends on an SL (the XBH), PA 5 is the rematch (FF, CU, FF), PA 9 ends on a CU
    rows = [
        (1, 1, "SL", "double", 1.25), (5, 1, "FF", None, np.nan), (5, 2, "CU", None, np.nan),
        (5, 3, "FF", "field_out", 0.0), (9, 1, "CU", "single", 0.9),
    ]
    df = pd.DataFrame(rows, columns=["at_bat_number", "pitch_number", "pitch_type", "events", "woba_value"])
    return df.assign(
        game_pk=1, pitcher=10, season=2021, game_date=pd.Timestamp("2021-05-01"), stand="R", p_throws="R",
        balls=0, strikes=0, outs_when_up=[0, 1, 1, 1, 2], runners_n=[0, 1, 1, 0, 0], risp=0, score_diff=0,
    )


def _usage():
    return pd.DataFrame({"pitcher": [10] * 3, "season": [2021] * 3, "pitch_type": ["FF", "CU", "SL"],
                         "season_usage_rate": [0.5, 0.3, 0.2]})


def _xbh():
    return pd.DataFrame({
        "game_pk": [1], "pitcher": [10], "season": [2021], "game_date": ["2021-05-01"], "next_at_bat_number": [5.0],
        "hit_pitch_type": ["SL"], "reused_same_type": [0.0],
    })


def _tables():
    pitches = add_platoon_match(_pitches())
    decisive = build_decisive_table(pitches, _usage())
    events = build_rematch_events(_xbh(), decisive, _usage(), pitches)
    return pitches, decisive, events


def test_platoon_match_is_missing_when_either_hand_is_unknown():
    df = pd.DataFrame({"stand": ["L", "R", None], "p_throws": ["R", "R", "L"]})
    out = add_platoon_match(df)["platoon_match"]
    assert out.iloc[0] == 0.0 and out.iloc[1] == 1.0 and np.isnan(out.iloc[2])


def test_decisive_table_keeps_only_pa_ending_pitches_with_their_season_usage():
    _, decisive, _ = _tables()
    assert decisive["at_bat_number"].tolist() == [1, 5, 9]
    assert np.allclose(decisive["season_usage_rate"], [0.2, 0.5, 0.3])
    assert str(decisive["pitch_type"].dtype) == "category"


def test_rematch_events_carry_the_hit_types_usage_and_the_pa_start_state():
    _, _, events = _tables()
    row = events.iloc[0]
    assert len(events) == 1 and row["at_bat_number"] == 5 and row["avoided"] == 1
    assert np.isclose(row["hit_usage_season"], 0.2) and row["hit_pitch_family"] == "breaking"
    # the PA ended with no runners, but it started with one on
    assert row["runners_n"] == 0 and row["runners_n_start"] == 1
    assert row["pitcher_id"] == "10"


def test_pa_pitches_are_every_typed_pitch_of_the_rematch_pa_only():
    pitches, decisive, events = _tables()
    pa = build_pa_pitches(events, pitches, _usage(), decisive)
    assert pa["pitch_type"].astype(str).tolist() == ["FF", "CU", "FF"]
    assert pa["pitch_type"].dtype == decisive["pitch_type"].dtype


def test_score_pa_averages_the_swap_difference_over_the_pa():
    pitches, decisive, events = _tables()
    pa = build_pa_pitches(events, pitches, _usage(), decisive)
    scored = score_pa(pa, _Stub(), FEATURES_SEASON, "hit_usage_season", "season_usage_rate")
    assert len(scored) == 1
    assert np.isclose(scored.loc[0, "selection_score"], 0.3)  # every pitch avoided the SL: 0.5 - 0.2
    assert np.isclose(scored.loc[0, "pred_reuse"], 0.5) and np.isclose(scored.loc[0, "pred_actual"], 0.2)


def test_group_gap_is_avoided_mean_minus_reused_mean():
    assert np.isclose(group_gap(np.array([0.0, 0.2, 1.0, 0.6]), np.array([0, 0, 1, 1])), 0.7)


def test_permutation_test_puts_perfectly_separated_groups_in_the_tail():
    values, avoided = np.array([0.0] * 5 + [1.0] * 5), np.array([0] * 5 + [1] * 5)
    result = permutation_test_gap(values, avoided, n_perm=500, seed=1)
    assert np.isclose(result["observed_gap"], 1.0) and result["p_value"] < 0.05


def test_refit_gap_draws_returns_one_finite_gap_per_refit():
    rng = np.random.default_rng(0)
    n = 60
    train = pd.DataFrame({
        "pitch_type": pd.Categorical(rng.choice(["FF", "SL"], size=n), categories=["FF", "SL", "CU"]),
        "stand": pd.Categorical(["R"] * n), "p_throws": pd.Categorical(["R"] * n),
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n), "outs_when_up": rng.integers(0, 3, n),
        "runners_n": rng.integers(0, 4, n), "risp": rng.integers(0, 2, n), "score_diff": rng.integers(-3, 3, n),
        "season_usage_rate": rng.uniform(0.1, 0.9, n), "woba_value": rng.uniform(0, 1, n),
    })
    pa = pd.DataFrame({
        "game_pk": [1, 1, 2, 2], "at_bat_number": [5, 5, 7, 7],
        "pitch_type": pd.Categorical(["FF", "SL", "FF", "FF"], categories=["FF", "SL", "CU"]),
        "hit_pitch_type": ["SL", "SL", "FF", "FF"], "season_usage_rate": [0.5] * 4, "hit_usage_season": [0.3, 0.3, 0.2, 0.2],
        "stand": pd.Categorical(["R"] * 4), "p_throws": pd.Categorical(["R"] * 4),
        "balls": [0, 1, 0, 1], "strikes": [0, 1, 0, 1], "outs_when_up": [0, 0, 1, 1],
        "runners_n": [0] * 4, "risp": [0] * 4, "score_diff": [0] * 4,
    })
    events = pd.DataFrame({"game_pk": [1, 2], "at_bat_number": [5, 7], "avoided": [1, 0]})
    draws = refit_gap_draws(train, pa, events, n_boot=3, seed=0)
    assert len(draws) == 3 and np.isfinite(draws).all()
