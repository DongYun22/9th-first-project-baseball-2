import numpy as np
import pandas as pd
import pytest

from src.analysis.avoidance_conditions import (
    add_prior_pitch_features,
    estimate_shrinkage_k,
    holm_adjust,
)


def _pitches():
    # pitcher 1: day1 -> 8 FF + 2 SL (SL decisive rvae +0.4, +0.2); day2 -> 5 FF + 5 CH; day3 = event day
    rows = []
    for d, pt, n, rv in [
        ("2024-04-01", "FF", 8, None), ("2024-04-01", "SL", 2, [0.4, 0.2]),
        ("2024-04-02", "FF", 5, None), ("2024-04-02", "CH", 5, [-0.1, -0.3, -0.2, 0.0, -0.2]),
        ("2024-04-03", "FF", 50, None),  # same day as event: must be excluded
    ]:
        for i in range(n):
            rows.append({"pitcher": 1, "game_date": pd.Timestamp(d), "pitch_type": pt,
                         "rvae": (rv[i] if rv is not None else np.nan)})
    return pd.DataFrame(rows)


def _events():
    return pd.DataFrame({"pitcher": [1], "game_date": [pd.Timestamp("2024-04-03")], "hit_pitch_type": ["FF"]})


def test_prior_features_use_only_strictly_earlier_dates():
    out = add_prior_pitch_features(_events(), _pitches(), shrinkage_k=2.0, min_alt_usage=0.05, min_prior_pitches=10)
    r = out.iloc[0]
    assert r["n_prior_pitches"] == 20  # FF 13 + SL 2 + CH 5; the 50 same-day FF are excluded
    assert r["hit_prior_usage"] == pytest.approx(13 / 20)
    assert r["hit_is_primary"] == 1.0
    assert r["n_alternatives"] == 2


def test_alternative_quality_is_shrunk_and_uses_best_alternative():
    out = add_prior_pitch_features(_events(), _pitches(), shrinkage_k=2.0, min_alt_usage=0.05, min_prior_pitches=10)
    r = out.iloc[0]
    # SL: sum .6 over 2 decisive -> .6/(2+2)=.15 ; CH: sum -.8 over 5 -> -.8/7
    assert r["alt_rvae_best"] == pytest.approx(-0.8 / 7)
    expected_w = (0.15 * 2 + (-0.8 / 7) * 5) / 7
    assert r["alt_rvae_wmean"] == pytest.approx(expected_w)


def test_event_with_too_little_history_gets_nan():
    out = add_prior_pitch_features(_events(), _pitches(), shrinkage_k=2.0, min_alt_usage=0.05, min_prior_pitches=500)
    assert out["hit_prior_usage"].isna().all() and out["alt_rvae_best"].isna().all()


def test_alternative_below_min_usage_is_ignored():
    out = add_prior_pitch_features(_events(), _pitches(), shrinkage_k=2.0, min_alt_usage=0.2, min_prior_pitches=10)
    assert out.iloc[0]["n_alternatives"] == 1  # SL (10%) dropped, CH (25%) kept


def test_estimate_shrinkage_k_recovers_ratio():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(400):
        mu = rng.normal(0, 0.1)
        for _ in range(40):
            rows.append({"pitcher": g, "pitch_type": "FF", "rvae": mu + rng.normal(0, 0.4)})
    k = estimate_shrinkage_k(pd.DataFrame(rows))
    assert k == pytest.approx(0.4**2 / 0.1**2, rel=0.25)


def test_holm_adjust():
    adj = holm_adjust([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
