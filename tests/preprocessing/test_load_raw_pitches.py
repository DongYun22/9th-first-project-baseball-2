import pandas as pd
import pytest

from src.preprocessing.load_raw_pitches import load_all_pitches


def test_load_all_pitches_concatenates_month_files_and_adds_season(tmp_path):
    (tmp_path / "2021").mkdir()
    (tmp_path / "2022").mkdir()
    pd.DataFrame({"game_date": ["2021-04-05"], "pitch_type": ["FF"]}).to_parquet(
        tmp_path / "2021" / "04.parquet", index=False
    )
    pd.DataFrame({"game_date": ["2022-05-10"], "pitch_type": ["SL"]}).to_parquet(
        tmp_path / "2022" / "05.parquet", index=False
    )

    combined = load_all_pitches(tmp_path)

    assert len(combined) == 2
    assert sorted(combined["season"].tolist()) == [2021, 2022]


def test_load_all_pitches_raises_when_no_files_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_pitches(tmp_path)
