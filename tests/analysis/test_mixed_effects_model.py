import numpy as np
import pandas as pd
import pytest

from src.analysis.mixed_effects_model import (
    MODEL_FORMULA_PLATOON,
    REQUIRED_MODEL_COLUMNS,
    build_combined_model_dataset,
    compare_random_effects,
    compute_calibration_table,
    compute_icc,
    rank_pitchers_by_group_slope,
)


def _event_row(**overrides):
    row = {
        "reused_same_type": 1.0, "baseline_usage": 0.3, "balls": 1, "strikes": 1,
        "outs_when_up": 0, "score_diff": 0, "stand": "R", "pitch_family": "fastball",
        "season": 2021, "catcher_changed": 0.0, "pitcher": 100, "pitcher_name": "A. Pitcher",
    }
    row.update(overrides)
    return row


def test_build_combined_model_dataset_concatenates_and_flags_group():
    xbh = pd.DataFrame([_event_row()])
    placebo = pd.DataFrame([_event_row(reused_same_type=0.0)])
    combined = build_combined_model_dataset(xbh, placebo)
    assert len(combined) == 2
    assert sorted(combined["group"].tolist()) == [0, 1]
    assert combined["pitcher"].tolist() == ["100", "100"]  # coerced to string


def test_build_combined_model_dataset_drops_rows_missing_required_covariates():
    xbh = pd.DataFrame([_event_row(), _event_row(baseline_usage=float("nan"))])
    placebo = pd.DataFrame([_event_row(reused_same_type=0.0)])
    combined = build_combined_model_dataset(xbh, placebo)
    assert len(combined) == 2  # the NaN baseline_usage row is dropped
    assert combined["baseline_usage"].notna().all()


def test_build_combined_model_dataset_accepts_extra_required_columns():
    xbh = pd.DataFrame([_event_row(platoon_match=1.0), _event_row(platoon_match=float("nan"))])
    placebo = pd.DataFrame([_event_row(platoon_match=0.0)])
    combined = build_combined_model_dataset(
        xbh, placebo, required_columns=REQUIRED_MODEL_COLUMNS + ["platoon_match"]
    )
    assert len(combined) == 2  # the NaN platoon_match row is dropped only when it is required
    default = build_combined_model_dataset(xbh, placebo)
    assert len(default) == 3


def test_platoon_formula_adds_main_effect_and_group_interaction_to_the_robustness_spec():
    assert "platoon_match" in MODEL_FORMULA_PLATOON
    assert "group:platoon_match" in MODEL_FORMULA_PLATOON
    assert "(0 + group | pitcher)" in MODEL_FORMULA_PLATOON
    assert "catcher_changed" not in MODEL_FORMULA_PLATOON


def test_compare_random_effects_correlates_only_pitchers_present_in_both():
    a = pd.DataFrame({"pitcher": ["1", "2", "3", "4"], "re_group": [0.1, 0.2, 0.3, 0.9]})
    b = pd.DataFrame({"pitcher": ["1", "2", "3", "5"], "re_group": [0.2, 0.4, 0.6, -9.0]})
    result = compare_random_effects(a, b)
    assert result["n_common"] == 3
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)


def test_compute_icc_known_value():
    icc = compute_icc(pitcher_intercept_variance=1.0)
    assert icc == pytest.approx(1 / (1 + (np.pi**2) / 3), abs=1e-6)


def test_compute_icc_is_zero_when_no_pitcher_variance():
    assert compute_icc(0.0) == 0.0


def test_rank_pitchers_by_group_slope_returns_extremes_in_order():
    random_effects = pd.DataFrame({
        "pitcher": [str(i) for i in range(1, 16)],
        "re_intercept": [0.0] * 15,
        "re_group": list(range(-7, 8)),  # -7..7
    })
    pitcher_meta = pd.DataFrame({
        "pitcher": [str(i) for i in range(1, 16)],
        "pitcher_name": [f"Pitcher {i}" for i in range(1, 16)],
        "n_xbh_events": [10] * 15,
    })
    top, bottom = rank_pitchers_by_group_slope(random_effects, pitcher_meta, n=3)
    assert top["re_group"].tolist() == [7, 6, 5]
    assert bottom["re_group"].tolist() == [-7, -6, -5]
    assert top.iloc[0]["pitcher_name"] == "Pitcher 15"


def test_compute_calibration_table_separates_low_and_high_bins():
    predicted = pd.Series(np.linspace(0.01, 0.99, 100))
    observed = pd.Series((predicted.to_numpy() > 0.5).astype(float))
    table = compute_calibration_table(observed, predicted, n_bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 100
    assert table.iloc[0]["mean_observed"] < table.iloc[-1]["mean_observed"]
