import numpy as np
import pandas as pd
import pytest

from src.analysis.selection_score import (
    build_pregame_counts,
    common_support,
    conditional_difference,
    expand_by_pitcher,
    lookup_pregame_usage,
    selection_score_parts,
    smoothed_usage,
    standardized_group_curves,
    tune_smoothing_k,
)


def _pitches():
    # pitcher 1: 2021 season (FF x3, SL x1), then 2022 with two game dates
    rows = [
        (1, 2021, "2021-04-01", "FF"), (1, 2021, "2021-04-01", "FF"),
        (1, 2021, "2021-04-01", "FF"), (1, 2021, "2021-04-01", "SL"),
        (1, 2022, "2022-04-01", "FF"), (1, 2022, "2022-04-01", "SL"),
        (1, 2022, "2022-04-08", "SL"), (1, 2022, "2022-04-08", "SL"),
    ]
    df = pd.DataFrame(rows, columns=["pitcher", "season", "game_date", "pitch_type"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def test_pregame_counts_exclude_the_game_itself():
    grid = build_pregame_counts(_pitches())
    row = grid[(grid["season"] == 2022) & (grid["game_date"] == "2022-04-08") & (grid["pitch_type"] == "SL")].iloc[0]
    # before 04-08 this season: one FF and one SL on 04-01 -- the two SL thrown on 04-08 must not count
    assert row["n_prior"] == 1 and row["total_prior"] == 2
    assert row["n_game"] == 2 and row["total_game"] == 2
    assert row["has_prev_season"] and np.isclose(row["prev_share"], 0.25)


def test_smoothed_usage_shrinks_toward_previous_season_and_keeps_missing_missing():
    grid = build_pregame_counts(_pitches())
    usage = smoothed_usage(grid, smoothing_k=2, min_prior_pitches=3)
    sl_0408 = usage[(grid["season"] == 2022) & (grid["game_date"] == "2022-04-08") & (grid["pitch_type"] == "SL")].iloc[0]
    assert np.isclose(sl_0408, (1 + 2 * 0.25) / (2 + 2))
    # first game of 2022: no season-to-date pitches, so the estimate is exactly last season's share
    sl_0401 = usage[(grid["season"] == 2022) & (grid["game_date"] == "2022-04-01") & (grid["pitch_type"] == "SL")].iloc[0]
    assert np.isclose(sl_0401, 0.25)
    # 2021 has no previous season and no prior pitches -> missing, never 0
    assert usage[grid["season"] == 2021].isna().all()


def test_lookup_pregame_usage_distinguishes_never_thrown_from_no_history():
    grid = build_pregame_counts(_pitches())
    grid["usage"] = smoothed_usage(grid, smoothing_k=2, min_prior_pitches=3)
    queries = pd.DataFrame({
        "pitcher": [1, 1, 1], "season": [2022, 2022, 2021],
        "game_date": pd.to_datetime(["2022-04-08", "2022-04-08", "2021-04-01"]),
        "pitch_type": ["SL", "CH", "CH"],
    })
    out = lookup_pregame_usage(queries, grid, min_prior_pitches=3)
    assert np.isclose(out.iloc[0], 0.375)
    assert out.iloc[1] == 0.0  # history exists and CH never appears in it -> a real zero
    assert np.isnan(out.iloc[2])  # no history at all -> missing


def test_tune_smoothing_k_picks_the_grid_value_with_lowest_error():
    grid = build_pregame_counts(_pitches())
    best_k, table = tune_smoothing_k(grid, seasons=[2022], candidates=[1, 1000], min_prior_pitches=3)
    assert set(table["smoothing_k"]) == {1, 1000}
    assert best_k == table.sort_values("weighted_mse").iloc[0]["smoothing_k"]


def test_conditional_difference_includes_the_covariance_term():
    delta, se = conditional_difference(
        usage_centered=np.array([0.0, 0.5]), beta_main=1.0, beta_interaction=-2.0,
        var_main=0.04, var_interaction=0.16, cov_main_interaction=-0.05,
    )
    assert np.allclose(delta, [1.0, 0.0])
    assert np.allclose(se, [0.2, np.sqrt(0.04 + 0.25 * 0.16 + 2 * 0.5 * -0.05)])


def test_standardized_group_curves_difference_equals_conditional_difference():
    terms = ["(Intercept)", "avoided", "u", "x", "avoided:u"]
    coef = pd.Series([0.1, 1.0, 0.5, 2.0, -2.0], index=terms)
    vcov = pd.DataFrame(np.diag([0.01, 0.04, 0.09, 0.01, 0.16]), index=terms, columns=terms)
    means = pd.Series([1.0, 0.4, 0.0, 3.0, 0.0], index=terms)
    curves = standardized_group_curves(coef, vcov, means, np.array([0.0, 0.5]), "avoided", "u", "avoided:u")
    reused, avoided = curves[curves["avoided"] == 0], curves[curves["avoided"] == 1]
    assert np.allclose(reused["predicted"], [0.1 + 6.0, 0.1 + 0.25 + 6.0])
    assert np.allclose(avoided["predicted"].to_numpy() - reused["predicted"].to_numpy(), [1.0, 0.0])
    assert (curves["se"] > 0).all()


def test_common_support_is_the_overlap_of_both_groups_central_ranges():
    lo, hi = common_support(np.linspace(0.0, 0.4, 101), np.linspace(0.2, 0.8, 101), lower_q=0.0, upper_q=1.0)
    assert np.isclose(lo, 0.2) and np.isclose(hi, 0.4)


def test_expand_by_pitcher_gives_duplicated_pitchers_distinct_replicate_ids():
    index_map = {"a": np.array([0, 1]), "b": np.array([2])}
    rows, replicate = expand_by_pitcher(index_map, ["a", "a", "b", "c"])
    assert rows.tolist() == [0, 1, 0, 1, 2]
    assert replicate.tolist() == [0, 0, 1, 1, 2]  # "c" has no rows; the two draws of "a" stay separate


class _Stub:
    def predict(self, X):
        base = np.where(X["pitch_type"].astype(str) == "SL", 0.5, 0.2)
        return base + (X["usage"].to_numpy() if "usage" in X else 0.0)


def test_selection_score_parts_swaps_type_and_its_usage_together():
    dtype = pd.CategoricalDtype(["SL", "FF"])
    df = pd.DataFrame({
        "pitch_type": pd.Categorical(["FF", "SL"], dtype=dtype), "usage": [0.1, 0.3],
        "hit_pitch_type": ["SL", "SL"], "hit_usage": [0.3, 0.3],
    })
    cf, actual = selection_score_parts(df, _Stub(), ["pitch_type", "usage"], "hit_pitch_type", "hit_usage", "usage")
    assert np.allclose(actual, [0.3, 0.8]) and np.allclose(cf, [0.8, 0.8])
    cf2, actual2 = selection_score_parts(df, _Stub(), ["pitch_type"], "hit_pitch_type", None, None)
    assert np.allclose(cf2 - actual2, [0.3, 0.0])


def test_selection_score_parts_rejects_a_usage_swap_without_its_column():
    df = pd.DataFrame({"pitch_type": pd.Categorical(["FF"]), "usage": [0.1], "hit_pitch_type": ["FF"]})
    with pytest.raises(ValueError):
        selection_score_parts(df, _Stub(), ["pitch_type", "usage"], "hit_pitch_type", None, "usage")
