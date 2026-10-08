"""Load and concatenate every collected Statcast month file."""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_all_pitches(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    month_files = sorted(raw_data_dir.glob("*/*.parquet"))
    if not month_files:
        raise FileNotFoundError(
            f"'{raw_data_dir}' 아래에 수집된 parquet 파일이 없습니다. "
            "먼저 src/collection/statcast_scraper.py를 실행하세요."
        )
    combined = pd.concat((pd.read_parquet(path) for path in month_files), ignore_index=True)
    combined["season"] = pd.to_datetime(combined["game_date"]).dt.year
    return combined
