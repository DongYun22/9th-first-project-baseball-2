"""Restrict the raw Statcast pitches to the team's shared research sample.

The team analyses one fixed sample: the pitches in the `statcast_<year>_min*_research.csv`
files (pitcher-seasons with >= 500 pitches, 2026 >= 300). Those CSVs have no catcher column
(score columns were added 2026-10-01), so the analysis still loads `data/raw` and keeps only
the rows whose key `(game_pk, at_bat_number, pitch_number, pitcher)` appears in the CSVs.
`pitch_type` is taken from the CSV so a later Statcast reclassification in `data/raw` cannot
make our sample differ from the teammates' one. Every other shared column is identical in
both sources.

Put the CSVs in `data/research/` (git-ignored) or point `RESEARCH_SAMPLE_DIR` at them. Only
the key columns and `pitch_type` are read; they are cached in `data/processed/` and re-read
when a CSV changes.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from src.preprocessing.load_raw_pitches import RAW_DATA_DIR, load_all_pitches

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SAMPLE_DIR = Path(os.environ.get("RESEARCH_SAMPLE_DIR", PROJECT_ROOT / "data" / "research"))
INDEX_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "research_sample_index.parquet"
CSV_GLOB = "statcast_*_research.csv"
KEY_COLUMNS = ["game_pk", "at_bat_number", "pitch_number", "pitcher"]
INDEX_COLUMNS = KEY_COLUMNS + ["pitch_type"]


def list_research_csvs(csv_dir: Path = RESEARCH_SAMPLE_DIR) -> list[Path]:
    csv_files = sorted(Path(csv_dir).glob(CSV_GLOB))
    misnamed = sorted(set(Path(csv_dir).glob("statcast_*.csv")) - set(csv_files))
    if misnamed:
        raise ValueError(
            f"이름이 {CSV_GLOB} 형식이 아니라 무시될 CSV가 있습니다: {[path.name for path in misnamed]}. "
            "다운로드하며 붙은 ' (1)' 같은 접미사를 떼고 원래 파일명으로 저장하세요 (연도 하나가 조용히 빠지는 것을 막기 위한 검사입니다)."
        )
    if not csv_files:
        raise FileNotFoundError(
            f"'{csv_dir}' 아래에 {CSV_GLOB} 파일이 없습니다. 팀 공유 research CSV를 data/research/ 에 넣거나 "
            "RESEARCH_SAMPLE_DIR 환경변수로 폴더를 지정하세요."
        )
    return csv_files


def read_research_index(csv_files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, usecols=INDEX_COLUMNS, encoding="utf-8-sig", low_memory=False) for path in csv_files]
    index = pd.concat(frames, ignore_index=True)
    if index.duplicated(KEY_COLUMNS).any():
        raise ValueError("research CSV에 (game_pk, at_bat_number, pitch_number, pitcher)가 중복된 행이 있습니다.")
    return index


def load_research_index(
    csv_dir: Path = RESEARCH_SAMPLE_DIR, cache_path: Path = INDEX_CACHE_PATH
) -> pd.DataFrame:
    csv_files = list_research_csvs(csv_dir)
    newest_csv = max(path.stat().st_mtime for path in csv_files)
    if cache_path.exists() and cache_path.stat().st_mtime >= newest_csv:
        return pd.read_parquet(cache_path)

    index = read_research_index(csv_files)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(cache_path, index=False)
    logger.info("research 표본 인덱스 %d행 생성 (CSV %d개) -> %s", len(index), len(csv_files), cache_path)
    return index


def restrict_to_research_sample(pitches: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Keep the pitches whose key is in `index`; `pitch_type` comes from `index`."""
    sample = pitches.drop(columns="pitch_type").merge(index, on=KEY_COLUMNS, how="inner", validate="many_to_one")
    if len(sample) != len(index):
        raise ValueError(
            f"research CSV {len(index)}행 중 {len(index) - len(sample)}행이 data/raw에 없습니다. "
            "data/raw를 다시 수집했는지(2021-03 ~ CSV 마지막 날짜) 확인하세요."
        )
    return sample


def load_research_pitches(
    raw_data_dir: Path = RAW_DATA_DIR,
    csv_dir: Path = RESEARCH_SAMPLE_DIR,
    cache_path: Path = INDEX_CACHE_PATH,
) -> pd.DataFrame:
    """All raw columns for exactly the pitches in the team's research CSVs."""
    index = load_research_index(csv_dir, cache_path)
    return restrict_to_research_sample(load_all_pitches(raw_data_dir), index)
