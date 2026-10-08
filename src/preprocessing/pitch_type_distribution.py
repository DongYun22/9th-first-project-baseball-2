"""Pitch-type and pitch-family frequency/distribution reporting."""

import pandas as pd

from src.preprocessing.pitch_family import CUTTER_CODE, map_pitch_family


def pitch_type_frequency(pitches: pd.DataFrame) -> pd.DataFrame:
    counts = pitches["pitch_type"].value_counts(dropna=False)
    share = counts / counts.sum()
    return pd.DataFrame({"count": counts, "share": share})


def pitch_family_frequency(pitches: pd.DataFrame) -> pd.DataFrame:
    """Fastball/breaking/offspeed counts and shares, plus the cutter's share
    of all pitches reported as its own row so FC's ambiguous status stays
    visible even though it is counted inside "fastball".
    """
    families = pitches["pitch_type"].map(map_pitch_family)
    counts = families.value_counts(dropna=False)
    total = counts.sum()
    result = pd.DataFrame({"count": counts, "share": counts / total})

    cutter_count = int((pitches["pitch_type"] == CUTTER_CODE).sum())
    result.loc["fastball_cutter_only"] = [cutter_count, cutter_count / total if total else float("nan")]
    return result


def family_distribution_by_season(pitches: pd.DataFrame) -> pd.DataFrame:
    working = pitches.copy()
    working["pitch_family"] = working["pitch_type"].map(map_pitch_family)
    working["season"] = pd.to_datetime(working["game_date"]).dt.year
    return pd.crosstab(working["season"], working["pitch_family"], normalize="index")
