import numpy as np
import pandas as pd

from src.analysis.pitcher_tendency import omega_squared, pitcher_rate, season_rate


def _events():
    return pd.DataFrame({
        "pitcher": [1, 1, 1, 2, 2], "season": [2021, 2021, 2022, 2021, 2021],
        "reused_same_type": [1, 0, 0, 1, 1],
    })


def test_pitcher_rate_is_one_minus_the_reuse_share_over_all_seasons():
    rates = pitcher_rate(_events()).set_index("pitcher")
    assert rates.loc[1, "n_events"] == 3 and np.isclose(rates.loc[1, "avoidance_rate"], 2 / 3)
    assert rates.loc[2, "n_events"] == 2 and np.isclose(rates.loc[2, "avoidance_rate"], 0.0)


def test_season_rate_splits_the_same_pitcher_by_season():
    rates = season_rate(_events()).set_index(["pitcher", "season"])
    assert np.isclose(rates.loc[(1, 2021), "avoidance_rate"], 0.5)
    assert np.isclose(rates.loc[(1, 2022), "avoidance_rate"], 1.0)


def test_omega_squared_is_one_when_every_pitcher_repeats_their_own_rate():
    stable = pd.DataFrame({"pitcher": [1, 1, 2, 2], "rate": [0.2, 0.2, 0.8, 0.8]})
    assert np.isclose(omega_squared(stable, "pitcher", "rate"), 1.0)


def test_omega_squared_goes_negative_for_pure_within_pitcher_noise():
    noisy = pd.DataFrame({"pitcher": [1, 1, 2, 2], "rate": [0.2, 0.8, 0.2, 0.8]})
    assert np.isclose(omega_squared(noisy, "pitcher", "rate"), -1 / 3)


def test_omega_squared_is_undefined_without_repeated_seasons():
    single = pd.DataFrame({"pitcher": [1, 2], "rate": [0.2, 0.8]})
    assert np.isnan(omega_squared(single, "pitcher", "rate"))
