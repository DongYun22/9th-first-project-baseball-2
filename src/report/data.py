"""Loads the team's research-sample CSVs once, with every column the final
report's analyses need and the situation columns derived from them.
"""

from __future__ import annotations

import pandas as pd

from src.analysis.hit_quality import XWOBA
from src.analysis.rematch_execution import MEASURES
from src.analysis.selection_value import add_platoon_match
from src.analysis.situation_matching import add_situation_columns
from src.preprocessing.research_sample import KEY_COLUMNS, list_research_csvs

LOAD_COLUMNS = [
    "game_pk", "game_date", "at_bat_number", "pitch_number", "pitcher", "player_name", "batter",
    "stand", "p_throws", "pitch_type", "events", "balls", "strikes", "outs_when_up",
    "inning", "inning_topbot", "on_1b", "on_2b", "on_3b", "home_score", "away_score",
    "woba_value", "woba_denom", "launch_speed", "launch_angle", XWOBA, *MEASURES,
]
EXPECTED_PITCHES = 3_592_302
EXPECTED_PITCHERS = 1_067


def load_pitches() -> pd.DataFrame:
    """Every pitch of the research sample (2021 ~ 2026-07-12) with `season`,
    the matching/situation columns and `platoon_match`.
    """
    frames = [
        pd.read_csv(path, usecols=LOAD_COLUMNS, encoding="utf-8-sig", low_memory=False)
        for path in list_research_csvs()
    ]
    pitches = pd.concat(frames, ignore_index=True)
    pitches["game_date"] = pd.to_datetime(pitches["game_date"])
    pitches["season"] = pitches["game_date"].dt.year
    if pitches.duplicated(KEY_COLUMNS).any():
        raise ValueError("research CSV에 중복 투구가 있습니다")
    return add_platoon_match(add_situation_columns(pitches))
