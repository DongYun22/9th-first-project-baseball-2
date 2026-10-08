import os

import pandas as pd
import pytest

from src.preprocessing.research_sample import (
    KEY_COLUMNS,
    load_research_index,
    load_research_pitches,
    restrict_to_research_sample,
)


def _pitches() -> pd.DataFrame:
    return pd.DataFrame({
        "game_pk": [1, 1, 1, 2],
        "at_bat_number": [1, 1, 2, 1],
        "pitch_number": [1, 2, 1, 1],
        "pitcher": [10, 10, 10, 20],
        "pitch_type": ["FF", "FF", "SL", "CH"],
        "home_score": [0, 0, 1, 0],
    })


def _write_csv(path, rows: pd.DataFrame) -> None:
    rows.to_csv(path, index=False, encoding="utf-8-sig")


def _write_raw(raw_dir, pitches: pd.DataFrame) -> None:
    (raw_dir / "2021").mkdir(parents=True)
    pitches.assign(game_date="2021-04-05").to_parquet(raw_dir / "2021" / "04.parquet", index=False)


def _csv_rows(pitches: pd.DataFrame, **overrides) -> pd.DataFrame:
    return pitches[KEY_COLUMNS + ["pitch_type"]].assign(game_type="R", **overrides)


def test_restrict_keeps_only_indexed_rows_and_keeps_extra_raw_columns():
    index = _pitches().iloc[[0, 2]][KEY_COLUMNS + ["pitch_type"]]

    sample = restrict_to_research_sample(_pitches(), index)

    assert len(sample) == 2
    assert sorted(sample["pitcher"]) == [10, 10]
    assert sorted(sample["home_score"]) == [0, 1]


def test_restrict_takes_pitch_type_from_the_index():
    index = _pitches().iloc[[0]][KEY_COLUMNS + ["pitch_type"]].assign(pitch_type="CH")

    sample = restrict_to_research_sample(_pitches(), index)

    assert sample["pitch_type"].tolist() == ["CH"]


def test_restrict_raises_when_csv_rows_are_missing_from_raw():
    index = pd.concat([
        _pitches()[KEY_COLUMNS + ["pitch_type"]],
        pd.DataFrame({"game_pk": [9], "at_bat_number": [1], "pitch_number": [1], "pitcher": [10], "pitch_type": ["FF"]}),
    ])

    with pytest.raises(ValueError, match="data/raw"):
        restrict_to_research_sample(_pitches(), index)


def test_load_research_index_raises_when_no_csv_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="research"):
        load_research_index(tmp_path, tmp_path / "cache.parquet")


def test_load_research_index_raises_on_misnamed_csv_instead_of_silently_skipping_it(tmp_path):
    rows = _csv_rows(_pitches())
    _write_csv(tmp_path / "statcast_2021_min500_research.csv", rows)
    _write_csv(tmp_path / "statcast_2026_min300_research (1).csv", rows)

    with pytest.raises(ValueError, match=r"\(1\)"):
        load_research_index(tmp_path, tmp_path / "cache.parquet")


def test_load_research_index_raises_on_duplicate_keys(tmp_path):
    rows = _pitches().iloc[[0, 0]]
    _write_csv(tmp_path / "statcast_2021_min500_research.csv", _csv_rows(rows))

    with pytest.raises(ValueError, match="중복"):
        load_research_index(tmp_path, tmp_path / "cache.parquet")


def test_load_research_pitches_end_to_end_with_bom_csv(tmp_path):
    raw_dir, csv_dir = tmp_path / "raw", tmp_path / "csv"
    csv_dir.mkdir()
    _write_raw(raw_dir, _pitches())
    _write_csv(csv_dir / "statcast_2021_min500_research.csv", _csv_rows(_pitches().iloc[:3]))

    sample = load_research_pitches(raw_dir, csv_dir, tmp_path / "cache.parquet")

    assert len(sample) == 3
    assert set(sample["pitcher"]) == {10}
    assert sample["season"].unique().tolist() == [2021]


def test_index_cache_is_rebuilt_when_a_csv_is_newer(tmp_path):
    csv_path = tmp_path / "statcast_2021_min500_research.csv"
    cache = tmp_path / "cache.parquet"
    _write_csv(csv_path, _csv_rows(_pitches().iloc[:2]))
    assert len(load_research_index(tmp_path, cache)) == 2

    _write_csv(csv_path, _csv_rows(_pitches()))
    newer = cache.stat().st_mtime + 10
    os.utime(csv_path, (newer, newer))

    assert len(load_research_index(tmp_path, cache)) == 4


def test_index_cache_is_reused_when_csvs_are_older(tmp_path):
    csv_path = tmp_path / "statcast_2021_min500_research.csv"
    cache = tmp_path / "cache.parquet"
    _write_csv(csv_path, _csv_rows(_pitches().iloc[:2]))
    load_research_index(tmp_path, cache)

    _write_csv(csv_path, _csv_rows(_pitches()))
    older = cache.stat().st_mtime - 10
    os.utime(csv_path, (older, older))

    assert len(load_research_index(tmp_path, cache)) == 2
