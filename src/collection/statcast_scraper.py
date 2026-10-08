"""Collect MLB Statcast pitch-by-pitch data one calendar month at a time.

Downloading the full 2021-2026 range in a single `pybaseball.statcast()`
call times out and hammers Baseball Savant, so this module walks the
range one in-season month at a time, writes each month to its own
parquet file, and skips any month whose file already exists -- a killed
or interrupted run can simply be re-launched and it resumes where it
left off.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import statcast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# 2021 시즌 개막(대략 4월) ~ 2026 올스타전 전날. game_type == "R" 필터가
# 스프링캠프/포스트시즌을 걸러내므로 매년 정확한 개막일을 하드코딩할 필요는
# 없고, 매년 3~10월(정규시즌이 있을 수 있는 달)만 순회하면 충분하다.
OVERALL_START_DATE = date(2021, 3, 1)
OVERALL_END_DATE = date(2026, 7, 14)
SEASON_MONTHS = range(3, 11)  # 3월~10월
SLEEP_SECONDS_BETWEEN_REQUESTS = 5

# Baseball Savant occasionally returns a malformed/empty CSV for a single
# day inside pybaseball's internal per-day parallel fetch, which raises
# instead of returning partial data. Retrying the whole month a few times
# with a backoff clears most of these transient hiccups without dying.
MAX_FETCH_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 30

COLUMNS = [
    "game_pk", "game_date", "pitcher", "player_name", "batter",
    "pitch_type", "events", "description",
    "at_bat_number", "pitch_number", "balls", "strikes", "outs_when_up",
    "inning", "inning_topbot", "stand", "p_throws",
    "home_score", "away_score", "post_home_score", "post_away_score",
    "zone", "plate_x", "plate_z", "pfx_x", "pfx_z",
    "release_speed", "release_spin_rate", "release_extension",
    "fielder_2",  # catcher's player id, used to detect a catcher change before the next at-bat
]


def month_windows(start_date: date, end_date: date) -> list[tuple[int, int, date, date]]:
    """Return (year, month, window_start, window_end) for each in-season month in range."""
    windows: list[tuple[int, int, date, date]] = []
    for year in range(start_date.year, end_date.year + 1):
        for month in SEASON_MONTHS:
            first_day = date(year, month, 1)
            last_day = date(year, month, calendar.monthrange(year, month)[1])
            window_start = max(first_day, start_date)
            window_end = min(last_day, end_date)
            if window_start > window_end:
                continue
            windows.append((year, month, window_start, window_end))
    return windows


def month_file_path(year: int, month: int) -> Path:
    return RAW_DATA_DIR / str(year) / f"{month:02d}.parquet"


def fetch_month(window_start: date, window_end: date) -> pd.DataFrame:
    raw = statcast(window_start.isoformat(), window_end.isoformat())
    if raw.empty:
        return raw.reindex(columns=COLUMNS)
    regular_season = raw[raw["game_type"] == "R"]
    return regular_season.reindex(columns=COLUMNS)


def fetch_month_with_retries(window_start: date, window_end: date) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            return fetch_month(window_start, window_end)
        except Exception as exc:  # noqa: BLE001 -- pybaseball raises various parser/network errors
            last_error = exc
            logger.warning(
                "%s ~ %s 수집 실패 (%d/%d회 시도): %s", window_start, window_end, attempt, MAX_FETCH_ATTEMPTS, exc
            )
            if attempt < MAX_FETCH_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)
    assert last_error is not None
    raise last_error


def collect_all(start_date: date = OVERALL_START_DATE, end_date: date = OVERALL_END_DATE) -> None:
    windows = month_windows(start_date, end_date)
    current_year: int | None = None
    year_row_count = 0

    for year, month, window_start, window_end in windows:
        if current_year is not None and year != current_year:
            logger.info("%d 시즌 수집 완료: %d건", current_year, year_row_count)
            year_row_count = 0
        current_year = year

        out_path = month_file_path(year, month)
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            logger.info("%d-%02d 이미 존재, 건너뜀 (%d건)", year, month, len(existing))
            year_row_count += len(existing)
            continue

        logger.info("%d-%02d 수집 중 (%s ~ %s)", year, month, window_start, window_end)
        try:
            month_df = fetch_month_with_retries(window_start, window_end)
        except Exception as exc:  # noqa: BLE001 -- keep the backfill going past a stuck month
            logger.error(
                "%d-%02d 수집 실패, 다음 달로 넘어감 (재실행하면 이 달만 재시도됨): %s", year, month, exc
            )
            time.sleep(SLEEP_SECONDS_BETWEEN_REQUESTS)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        month_df.to_parquet(out_path, index=False)
        logger.info("%d-%02d 저장 완료: %d건 -> %s", year, month, len(month_df), out_path)
        year_row_count += len(month_df)

        time.sleep(SLEEP_SECONDS_BETWEEN_REQUESTS)

    if current_year is not None:
        logger.info("%d 시즌 수집 완료: %d건", current_year, year_row_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Statcast pitch data month by month.")
    parser.add_argument("--start-date", type=date.fromisoformat, default=OVERALL_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=OVERALL_END_DATE)
    args = parser.parse_args()
    collect_all(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
