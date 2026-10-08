from datetime import date

import pandas as pd

from src.collection import statcast_scraper


def test_month_windows_clips_first_and_last_month_to_overall_range():
    windows = statcast_scraper.month_windows(date(2021, 4, 15), date(2021, 6, 5))
    assert windows == [
        (2021, 4, date(2021, 4, 15), date(2021, 4, 30)),
        (2021, 5, date(2021, 5, 1), date(2021, 5, 31)),
        (2021, 6, date(2021, 6, 1), date(2021, 6, 5)),
    ]


def test_month_windows_skips_offseason_months_across_year_boundary():
    windows = statcast_scraper.month_windows(date(2021, 10, 25), date(2022, 4, 3))
    assert windows == [
        (2021, 10, date(2021, 10, 25), date(2021, 10, 31)),
        (2022, 3, date(2022, 3, 1), date(2022, 3, 31)),
        (2022, 4, date(2022, 4, 1), date(2022, 4, 3)),
    ]


def test_month_windows_caps_2026_at_july_14():
    windows = statcast_scraper.month_windows(date(2026, 6, 1), date(2026, 7, 14))
    assert windows == [
        (2026, 6, date(2026, 6, 1), date(2026, 6, 30)),
        (2026, 7, date(2026, 7, 1), date(2026, 7, 14)),
    ]


def test_month_file_path_is_zero_padded(tmp_path, monkeypatch):
    monkeypatch.setattr(statcast_scraper, "RAW_DATA_DIR", tmp_path)
    assert statcast_scraper.month_file_path(2021, 4) == tmp_path / "2021" / "04.parquet"


def test_collect_all_skips_months_whose_file_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(statcast_scraper, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(statcast_scraper, "SLEEP_SECONDS_BETWEEN_REQUESTS", 0)
    existing_path = tmp_path / "2021" / "04.parquet"
    existing_path.parent.mkdir(parents=True)
    pd.DataFrame({"pitch_type": ["FF", "SL"]}).to_parquet(existing_path, index=False)

    def fail_if_called(start_dt, end_dt):
        raise AssertionError("statcast() should not be called for a month already on disk")

    monkeypatch.setattr(statcast_scraper, "statcast", fail_if_called)

    statcast_scraper.collect_all(date(2021, 4, 1), date(2021, 4, 30))

    assert pd.read_parquet(existing_path).shape[0] == 2


def test_collect_all_fetches_and_saves_missing_month(tmp_path, monkeypatch):
    monkeypatch.setattr(statcast_scraper, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(statcast_scraper, "SLEEP_SECONDS_BETWEEN_REQUESTS", 0)

    fake_raw = pd.DataFrame(
        {
            "game_type": ["R", "R", "S"],
            **{col: [1, 2, 3] for col in statcast_scraper.COLUMNS if col != "game_type"},
        }
    )

    def fake_statcast(start_dt, end_dt):
        assert (start_dt, end_dt) == ("2021-04-01", "2021-04-30")
        return fake_raw

    monkeypatch.setattr(statcast_scraper, "statcast", fake_statcast)

    statcast_scraper.collect_all(date(2021, 4, 1), date(2021, 4, 30))

    saved = pd.read_parquet(tmp_path / "2021" / "04.parquet")
    assert len(saved) == 2  # the game_type == "S" (spring training) row is dropped
    assert list(saved.columns) == statcast_scraper.COLUMNS


def test_collect_all_retries_a_transient_failure_and_still_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(statcast_scraper, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(statcast_scraper, "SLEEP_SECONDS_BETWEEN_REQUESTS", 0)
    monkeypatch.setattr(statcast_scraper, "RETRY_BACKOFF_SECONDS", 0)

    fake_raw = pd.DataFrame(
        {"game_type": ["R"], **{col: [1] for col in statcast_scraper.COLUMNS if col != "game_type"}}
    )
    calls = {"count": 0}

    def flaky_statcast(start_dt, end_dt):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("Error tokenizing data. C error: Expected 1 fields in line 12, saw 2")
        return fake_raw

    monkeypatch.setattr(statcast_scraper, "statcast", flaky_statcast)

    statcast_scraper.collect_all(date(2021, 4, 1), date(2021, 4, 30))

    assert calls["count"] == 2
    assert (tmp_path / "2021" / "04.parquet").exists()


def test_collect_all_skips_forward_past_a_month_that_exhausts_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(statcast_scraper, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(statcast_scraper, "SLEEP_SECONDS_BETWEEN_REQUESTS", 0)
    monkeypatch.setattr(statcast_scraper, "RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(statcast_scraper, "MAX_FETCH_ATTEMPTS", 2)

    fake_raw = pd.DataFrame(
        {"game_type": ["R"], **{col: [1] for col in statcast_scraper.COLUMNS if col != "game_type"}}
    )

    def always_fails_for_april_only(start_dt, end_dt):
        if start_dt.startswith("2021-04"):
            raise ValueError("Error tokenizing data. C error: Expected 1 fields in line 12, saw 2")
        return fake_raw

    monkeypatch.setattr(statcast_scraper, "statcast", always_fails_for_april_only)

    statcast_scraper.collect_all(date(2021, 4, 1), date(2021, 5, 31))

    assert not (tmp_path / "2021" / "04.parquet").exists()
    assert (tmp_path / "2021" / "05.parquet").exists()
