import pandas as pd

from src.preprocessing.pitch_type_distribution import (
    family_distribution_by_season,
    pitch_family_frequency,
    pitch_type_frequency,
)


def _pitches():
    return pd.DataFrame(
        {
            "pitch_type": ["FF", "FF", "SL", "FC", "SL"],
            "game_date": ["2021-05-01", "2022-05-01", "2021-05-01", "2022-05-01", "2022-05-01"],
        }
    )
    # 2021: FF, SL (1 fastball, 1 breaking)
    # 2022: FF, FC, SL (2 fastball [FF, FC], 1 breaking [SL])


def test_pitch_type_frequency_counts_and_shares():
    result = pitch_type_frequency(_pitches())
    assert result.loc["FF", "count"] == 2
    assert result.loc["FF", "share"] == 2 / 5
    assert result.loc["SL", "count"] == 2


def test_pitch_family_frequency_groups_cutter_into_fastball_and_reports_its_share():
    result = pitch_family_frequency(_pitches())
    assert result.loc["fastball", "count"] == 3  # FF, FF, FC
    assert result.loc["breaking", "count"] == 2  # SL, SL
    assert result.loc["fastball_cutter_only", "count"] == 1  # FC
    assert result.loc["fastball_cutter_only", "share"] == 1 / 5


def test_family_distribution_by_season_pivots_share_by_year():
    table = family_distribution_by_season(_pitches())
    assert table.loc[2021, "fastball"] == 0.5
    assert table.loc[2021, "breaking"] == 0.5
    assert table.loc[2022, "fastball"] == 2 / 3
    assert table.loc[2022, "breaking"] == 1 / 3
