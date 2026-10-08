import numpy as np
import pandas as pd
import pytest

from src.analysis.avoidance_stats import (
    bootstrap_relative_reduction_ci,
    relative_reduction,
    summarize_reuse_rate_reduction,
    build_stratified_placebo_candidates,
    compute_expected_reuse_prob,
    compute_season_usage_rate,
    match_treatment_to_control,
    summarize_avoidance_by_level,
    summarize_diff_in_diff,
    summarize_paired_diff,
)


def test_compute_season_usage_rate_computes_share_per_pitcher_season_pitch_type():
    pitches = pd.DataFrame({
        "pitcher": [1, 1, 1, 1, 2, 2],
        "season": [2021, 2021, 2021, 2021, 2021, 2021],
        "pitch_type": ["FF", "FF", "SL", "SL", "FF", "SL"],
    })
    usage = compute_season_usage_rate(pitches)
    row = usage[(usage["pitcher"] == 1) & (usage["pitch_type"] == "FF")].iloc[0]
    assert row["n"] == 2
    assert row["total"] == 4
    assert row["season_usage_rate"] == 0.5


def test_compute_season_usage_rate_ignores_pitches_with_no_pitch_type():
    pitches = pd.DataFrame({
        "pitcher": [1, 1, 1, 1],
        "season": [2021, 2021, 2021, 2021],
        "pitch_type": ["FF", "SL", None, None],
    })
    usage = compute_season_usage_rate(pitches)
    assert usage["pitch_type"].notna().all()  # no bucket for "missing" that a missing event type could match
    ff = usage[usage["pitch_type"] == "FF"].iloc[0]
    assert ff["total"] == 2  # denominator counts only pitches with a known type
    assert ff["season_usage_rate"] == 0.5


def test_build_stratified_placebo_candidates_matches_quota_and_is_reproducible():
    treatment = pd.DataFrame({
        "season": [2021, 2021, 2022],
        "pitch_family": ["fastball", "breaking", "fastball"],
    })  # quota: (2021,fastball)=1, (2021,breaking)=1, (2022,fastball)=1
    pitches = pd.DataFrame({
        "events": ["field_out"] * 10,
        "pitch_type": ["FF", "FF", "FF", "SL", "SL", "SL", "FF", "FF", "FF", "FF"],
        "season": [2021] * 6 + [2022] * 4,
    })

    sample1 = build_stratified_placebo_candidates(pitches, treatment, {"field_out"}, seed=1)
    counts = sample1.groupby(["season", "pitch_family"]).size()
    assert counts[(2021, "fastball")] == 1
    assert counts[(2021, "breaking")] == 1
    assert counts[(2022, "fastball")] == 1

    sample2 = build_stratified_placebo_candidates(pitches, treatment, {"field_out"}, seed=1)
    pd.testing.assert_frame_equal(sample1.sort_index(), sample2.sort_index())


def test_build_stratified_placebo_candidates_takes_all_when_pool_smaller_than_quota():
    treatment = pd.DataFrame({
        "season": [2021, 2021, 2021],
        "pitch_family": ["fastball", "fastball", "fastball"],
    })  # quota: (2021, fastball) = 3
    pitches = pd.DataFrame({
        "events": ["field_out", "field_out"],
        "pitch_type": ["FF", "FF"],
        "season": [2021, 2021],
    })  # only 2 candidates available
    sample = build_stratified_placebo_candidates(pitches, treatment, {"field_out"}, seed=1)
    assert len(sample) == 2


def test_build_stratified_placebo_candidates_applies_candidate_filter_before_sampling():
    treatment = pd.DataFrame({
        "season": [2021, 2021],
        "pitch_family": ["fastball", "fastball"],
    })  # quota: (2021, fastball) = 2
    pitches = pd.DataFrame({
        "events": ["field_out"] * 8,
        "pitch_type": ["FF"] * 8,
        "season": [2021] * 8,
        "eligible": [True, False, False, False, False, False, False, True],
    })

    def only_eligible(_pitches, candidates):
        return candidates[candidates["eligible"]]

    sample = build_stratified_placebo_candidates(
        pitches, treatment, {"field_out"}, seed=1, candidate_filter=only_eligible
    )
    assert len(sample) == 2
    assert sample["eligible"].all()


def test_build_stratified_placebo_candidates_can_match_on_pitcher_too():
    treatment = pd.DataFrame({
        "pitcher": [1, 1, 2],
        "season": [2021, 2021, 2021],
        "pitch_family": ["fastball", "fastball", "fastball"],
    })  # quota: pitcher 1 -> 2, pitcher 2 -> 1
    pitches = pd.DataFrame({
        "events": ["field_out"] * 6,
        "pitcher": [1, 1, 1, 2, 3, 3],  # pitcher 3 has no treatment events: never sampled
        "pitch_type": ["FF"] * 6,
        "season": [2021] * 6,
    })
    sample = build_stratified_placebo_candidates(
        pitches, treatment, {"field_out"}, seed=1, strata_cols=("pitcher", "season", "pitch_family")
    )
    assert sorted(sample["pitcher"].tolist()) == [1, 1, 2]


def test_pitcher_matching_takes_only_what_exists_when_a_pitcher_has_too_few_candidates():
    treatment = pd.DataFrame({
        "pitcher": [1, 1, 1],
        "season": [2021, 2021, 2021],
        "pitch_family": ["fastball"] * 3,
    })  # quota 3, but pitcher 1 has just one candidate; pitcher 9 has none but is never asked for
    pitches = pd.DataFrame({
        "events": ["field_out", "field_out"],
        "pitcher": [1, 9],
        "pitch_type": ["FF", "FF"],
        "season": [2021, 2021],
    })
    sample = build_stratified_placebo_candidates(
        pitches, treatment, {"field_out"}, seed=1, strata_cols=("pitcher", "season", "pitch_family")
    )
    assert sample["pitcher"].tolist() == [1]


def test_match_treatment_to_control_trims_treatment_to_the_controls_stratum_counts():
    strata = ("pitcher", "season", "pitch_family")
    treatment = pd.DataFrame({
        "pitcher": [1, 1, 1, 2, 3],
        "season": [2021] * 5,
        "pitch_family": ["fastball"] * 5,
        "id": [10, 11, 12, 20, 30],
    })
    control = pd.DataFrame({
        "pitcher": [1, 1, 2, 2],  # pitcher 1: 2 controls, pitcher 2: 2 controls, pitcher 3: none
        "season": [2021] * 4,
        "pitch_family": ["fastball"] * 4,
    })
    matched = match_treatment_to_control(treatment, control, strata, seed=1)
    counts = matched.groupby("pitcher").size().to_dict()
    assert counts == {1: 2, 2: 1}  # pitcher 1 trimmed 3->2, pitcher 2 kept (1 treatment), pitcher 3 dropped
    assert matched["id"].isin(treatment["id"]).all()


def _events(pitch_type, baselines, shares, reused, n_pitches=4):
    return pd.DataFrame({
        "hit_pitch_type": pitch_type,
        "has_next_ab": True,
        "season_usage_rate": baselines,
        "same_type_share": shares,
        "reused_same_type": reused,
        "next_ab_pitch_count": n_pitches,
    })


def test_summarize_avoidance_by_level_reports_net_drop_and_reuse_per_level_and_filters_small_ones():
    xbh = pd.concat([
        _events("FF", [0.4] * 4, [0.1, 0.3, 0.2, 0.2], [0, 1, 1, 0]),   # drop 0.2 on average
        _events("SL", [0.3] * 4, [0.0, 0.2, 0.1, 0.1], [0, 1, 0, 0]),
        _events("KN", [0.3] * 1, [0.0], [0]),                              # too few events -> dropped
    ])
    placebo = pd.concat([
        _events("FF", [0.4] * 4, [0.3, 0.5, 0.4, 0.4], [1, 1, 1, 0]),   # drop 0.0
        _events("SL", [0.3] * 4, [0.2, 0.4, 0.3, 0.3], [1, 1, 1, 0]),
        _events("KN", [0.3] * 1, [0.3], [1]),
    ])
    table = summarize_avoidance_by_level(xbh, placebo, by="hit_pitch_type", baseline_col="season_usage_rate", min_n=4)

    assert table["hit_pitch_type"].tolist() == ["FF", "SL"]
    ff = table[table["hit_pitch_type"] == "FF"].iloc[0]
    assert ff["xbh_drop"] == pytest.approx(0.2)
    assert ff["placebo_drop"] == pytest.approx(0.0)
    assert ff["net_drop"] == pytest.approx(0.2)
    assert ff["reuse_xbh"] == pytest.approx(0.5)
    assert ff["reuse_placebo"] == pytest.approx(0.75)
    assert ff["n_xbh"] == 4 and ff["n_placebo"] == 4
    assert ff["net_ci_low"] < ff["net_drop"] < ff["net_ci_high"]


def test_relative_reduction_applies_the_placebo_change_to_the_expected_usage():
    # placebo usage falls 0.40 -> 0.30, so without the hit the XBH group would have gone 0.40 -> 0.30
    assert relative_reduction(0.40, 0.20, 0.40, 0.30) == pytest.approx(1 - 0.20 / 0.30)
    # placebo usage rises 0.30 -> 0.40 (drop -0.10): expected post = 0.30 + 0.10
    assert relative_reduction(0.30, 0.20, 0.30, 0.40) == pytest.approx(1 - 0.20 / 0.40)


def test_bootstrap_relative_reduction_ci_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    xbh_base = rng.normal(0.40, 0.05, 300)
    xbh_post = xbh_base - rng.normal(0.15, 0.05, 300)
    placebo_base = rng.normal(0.40, 0.05, 300)
    placebo_post = placebo_base - rng.normal(0.02, 0.05, 300)
    low, high = bootstrap_relative_reduction_ci(xbh_base, xbh_post, placebo_base, placebo_post, n_boot=300, seed=1)
    point = relative_reduction(xbh_base.mean(), xbh_post.mean(), placebo_base.mean(), placebo_post.mean())
    assert low < point < high
    again = bootstrap_relative_reduction_ci(xbh_base, xbh_post, placebo_base, placebo_post, n_boot=300, seed=1)
    assert (low, high) == again


def test_summarize_reuse_rate_reduction_orders_by_requested_levels_and_adds_reuse_diff_ci():
    xbh = pd.concat([
        _events("SL", [0.3] * 4, [0.0, 0.2, 0.1, 0.1], [0, 1, 0, 0]),
        _events("FF", [0.4] * 4, [0.1, 0.3, 0.2, 0.2], [0, 1, 1, 0]),
    ])
    placebo = pd.concat([
        _events("SL", [0.3] * 4, [0.2, 0.4, 0.3, 0.3], [1, 1, 1, 0]),
        _events("FF", [0.4] * 4, [0.3, 0.5, 0.4, 0.4], [1, 1, 1, 0]),
    ])
    table = summarize_reuse_rate_reduction(
        xbh, placebo, by="hit_pitch_type", baseline_col="season_usage_rate", levels=["FF", "SL", "KN"], n_boot=50, seed=1
    )
    assert table["hit_pitch_type"].tolist() == ["FF", "SL"]  # requested order; missing level skipped
    ff = table.iloc[0]
    assert ff["xbh_baseline"] == pytest.approx(0.4) and ff["xbh_post"] == pytest.approx(0.2)
    assert ff["net_drop"] == pytest.approx(0.2)
    assert ff["relative_reduction"] == pytest.approx(1 - 0.2 / 0.4)  # placebo drop is 0
    assert ff["reuse_diff"] == pytest.approx(0.5 - 0.75)
    assert ff["reuse_diff_ci_low"] < ff["reuse_diff"] < ff["reuse_diff_ci_high"]
    assert ff["rel_ci_low"] <= ff["relative_reduction"] <= ff["rel_ci_high"]


def test_compute_expected_reuse_prob_formula():
    baseline = pd.Series([0.0, 0.5, 0.2])
    counts = pd.Series([3, 2, 5])
    result = compute_expected_reuse_prob(baseline, counts)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == pytest.approx(1 - 0.5**2)
    assert result.iloc[2] == pytest.approx(1 - 0.8**5)


def test_summarize_paired_diff_computes_mean_ci_and_pvalues():
    diff = pd.Series([-0.1, -0.12, -0.09, -0.11, -0.1] * 20)
    result = summarize_paired_diff(diff)
    assert result["mean"] == pytest.approx(-0.104, abs=0.001)
    assert result["ci_high"] < 0
    assert result["t_p"] < 0.001
    assert result["w_p"] < 0.001


def test_summarize_diff_in_diff_computes_net_effect_and_significant_p_value():
    d_treatment = pd.Series([0.5, 0.6, 0.55, 0.52, 0.58] * 20)  # mean 0.55
    d_control = pd.Series([0.1, 0.15, 0.12, 0.11, 0.13] * 20)  # mean 0.122
    result = summarize_diff_in_diff(d_treatment, d_control, alternative="greater")
    assert result["net_effect"] == pytest.approx(0.55 - 0.122, abs=0.001)
    assert result["ci_low"] > 0
    assert result["t_p"] < 0.001
    assert result["u_p"] < 0.001
