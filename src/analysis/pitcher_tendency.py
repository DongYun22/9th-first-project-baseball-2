"""Pitcher-level avoidance rates and how much of their spread is a stable
pitcher trait (final report p.11).

avoidance rate = 1 - mean(reused_same_type) over a pitcher's rematch events.
"""

from __future__ import annotations

import pandas as pd


def _agg_rate(events: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = events.groupby(group_cols)["reused_same_type"].agg(n_events="size", reuse_rate="mean").reset_index()
    out["avoidance_rate"] = 1 - out["reuse_rate"]
    return out.drop(columns="reuse_rate")


def season_rate(events: pd.DataFrame) -> pd.DataFrame:
    return _agg_rate(events, ["pitcher", "season"])


def pitcher_rate(events: pd.DataFrame) -> pd.DataFrame:
    return _agg_rate(events, ["pitcher"])


def omega_squared(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    """Bias-corrected share of variance between groups:
    (SS_between - (k-1) * MS_within) / (SS_total + MS_within).
    Eta-squared overstates this share when there are many groups with few
    rows each; omega-squared corrects that and can be negative for pure noise.
    """
    grand_mean = df[value_col].mean()
    n, k = len(df), df[group_col].nunique()
    total_ss = ((df[value_col] - grand_mean) ** 2).sum()
    if total_ss == 0 or n <= k:
        return float("nan")
    group_means = df.groupby(group_col)[value_col].transform("mean")
    between_ss = ((group_means - grand_mean) ** 2).sum()
    ms_within = (total_ss - between_ss) / (n - k)
    return float((between_ss - (k - 1) * ms_within) / (total_ss + ms_within))
