"""Step 4 of the final report (p.11): pitcher cases and how stable a
pitcher's avoidance rate is across seasons.
"""

from __future__ import annotations

import pandas as pd

from src.analysis.pitcher_tendency import omega_squared, pitcher_rate, season_rate

MIN_EVENTS = 10          # rematches needed for a pitcher to be listed as a case
MIN_SEASON_EVENTS = 5    # rematches needed for a pitcher-season to count toward stability
N_CASES = 5


def pitcher_cases(xbh: pd.DataFrame) -> pd.DataFrame:
    """Pitchers with at least MIN_EVENTS rematches, sorted by avoidance rate (highest first)."""
    rates = pitcher_rate(xbh)
    rates = rates[rates["n_events"] >= MIN_EVENTS].copy()
    names = xbh.drop_duplicates("pitcher").set_index("pitcher")["pitcher_name"]
    rates["pitcher_name"] = rates["pitcher"].map(names)
    return rates.sort_values("avoidance_rate", ascending=False).reset_index(drop=True)


def season_stability(xbh: pd.DataFrame) -> dict:
    """Omega-squared of pitcher-season avoidance rates across pitchers with two or more qualified seasons."""
    rates = season_rate(xbh)
    rates = rates[rates["n_events"] >= MIN_SEASON_EVENTS]
    multi = rates[rates.groupby("pitcher")["pitcher"].transform("size") >= 2]
    return {"omega_squared": omega_squared(multi, "pitcher", "avoidance_rate"),
            "n_pitchers": int(multi["pitcher"].nunique()), "n_pitcher_seasons": len(multi)}
