"""Step 2 of the final report (p.6-7): what kind of hit is avoided more.

p.6 -- result x contact quality (xwOBA quartile, barrel).
p.7 -- GBM on the change in the hit type's usage with SHAP importances
       (Beomseok's analysis; same-season rematches).
"""

from __future__ import annotations

import pandas as pd

from src.analysis.hit_quality import add_xwoba_quartile, build_batted_ball_events, cell_table
from src.analysis.usage_change_factors import build_design_matrix, build_same_season_rematch_events, fit_and_explain

# model input -> the name the report uses for it
FEATURE_LABELS = {
    "events_home_run": "홈런 여부", "score_diff_bat": "점수차", "launch_speed": "타구 속도",
    "pitch_group_SI": "싱커(SI)", "release_speed": "릴리스 구속", "matchup_R-L": "우투-좌타",
    "pfx_z": "수직 무브먼트", "matchup_L-R": "좌투-우타", "launch_angle": "발사각",
    "pfx_x_arm": "수평 무브먼트", "estimated_woba_using_speedangle": "xwOBA",
}


def hit_quality_tables(pitches: pd.DataFrame, season_usage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rematch batted balls split by result x xwOBA quartile (reference:
    weak out = 아웃 in Q1) and by result x barrel (reference: non-barrel out).
    `vs_ref` is the extra avoidance over the reference cell.
    """
    events = add_xwoba_quartile(build_batted_ball_events(pitches, season_usage, "same_batter"))
    cells = cell_table(events, "season_usage_rate")
    labelled = events.assign(barrel_label=events["barrel"].map({True: "배럴", False: "비배럴"}))
    barrel = cell_table(labelled, "season_usage_rate", by="barrel_label", ref=("아웃", "비배럴"))
    return cells, barrel


def shap_factor_tables(pitches: pd.DataFrame, season_usage: pd.DataFrame) -> dict:
    """GBM test R² and mean |SHAP| per input for the usage change after an XBH.
    The 80/20 split is random, so R² moves by about +-0.01 with the rows that land in the test set.
    """
    events = build_same_season_rematch_events(pitches, season_usage)
    X, y = build_design_matrix(events)
    result = fit_and_explain(X, y)
    result["importance"]["label"] = result["importance"]["feature"].map(FEATURE_LABELS).fillna(result["importance"]["feature"])
    result["n_events"] = len(events)
    return result
