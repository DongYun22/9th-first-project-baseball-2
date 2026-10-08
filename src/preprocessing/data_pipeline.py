"""End-to-end pipeline: load the team's research sample of Statcast pitches
(data/raw restricted to the rows in the research CSVs), report pitch-type /
pitch-family distributions, and build the next-at-bat pitch-reuse dataset.
"""

import logging
from pathlib import Path

from src.preprocessing.build_next_ab_dataset import build_event_dataset, identify_extra_base_hit_events
from src.preprocessing.research_sample import load_research_pitches
from src.preprocessing.pitch_type_distribution import (
    family_distribution_by_season,
    pitch_family_frequency,
    pitch_type_frequency,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_BASENAME = "pitch_reuse_after_xbh_events"


def run() -> None:
    pitches = load_research_pitches()
    logger.info("research 표본 투구 데이터 로드 완료: %d건 (투수 %d명)", len(pitches), pitches["pitcher"].nunique())

    xbh_pitches = identify_extra_base_hit_events(pitches)

    logger.info("=== 구종 분포: 전체 투구 ===\n%s", pitch_type_frequency(pitches))
    logger.info("=== 구종 분포: 장타 허용 투구 ===\n%s", pitch_type_frequency(xbh_pitches))
    logger.info("=== 구종 계열 분포: 전체 투구 ===\n%s", pitch_family_frequency(pitches))
    logger.info("=== 구종 계열 분포: 장타 허용 투구 ===\n%s", pitch_family_frequency(xbh_pitches))
    logger.info("=== 시즌별 구종 계열 비중: 전체 투구 ===\n%s", family_distribution_by_season(pitches))
    logger.info("=== 시즌별 구종 계열 비중: 장타 허용 투구 ===\n%s", family_distribution_by_season(xbh_pitches))

    event_dataset = build_event_dataset(pitches)

    total_events = len(event_dataset)
    has_next_ab_share = event_dataset["has_next_ab"].mean() if total_events else float("nan")
    season_counts = event_dataset["season"].value_counts().sort_index()
    logger.info("전체 장타 이벤트 수: %d", total_events)
    logger.info("has_next_ab=True 비율: %.1f%%", has_next_ab_share * 100)
    logger.info("시즌별 이벤트 수 분포:\n%s", season_counts)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DATA_DIR / f"{OUTPUT_BASENAME}.parquet"
    csv_path = PROCESSED_DATA_DIR / f"{OUTPUT_BASENAME}.csv"
    event_dataset.to_parquet(parquet_path, index=False)
    event_dataset.to_csv(csv_path, index=False)
    logger.info("최종 데이터셋 저장 완료: %s, %s", parquet_path, csv_path)


if __name__ == "__main__":
    run()
