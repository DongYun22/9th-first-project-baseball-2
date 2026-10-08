# 📂 Research Sample (팀 공유 분석 표본 CSV)

팀 전체가 같은 표본으로 분석하기 위한 CSV를 두는 폴더입니다. CSV 파일은 용량이 커서(약 850MB) `.gitignore` 대상이며, [공유 드라이브](https://drive.google.com/drive/folders/1nP5Z0f7q1cNftgKkVTsWe329ishi8iUM)에서 받아 이 폴더에 넣습니다.

## 파일

| 파일 | 투구 수 | 투수 | 기간 |
|---|---|---|---|
| `statcast_2021_min500_research.csv` | 634,829 | 486 | 2021-04-01 ~ 10-03 |
| `statcast_2022_min500_research.csv` | 637,264 | 473 | 2022-04-07 ~ 10-05 |
| `statcast_2023_min500_research.csv` | 648,341 | 479 | 2023-03-30 ~ 10-01 |
| `statcast_2024_min500_research.csv` | 641,086 | 474 | 2024-03-28 ~ 09-30 |
| `statcast_2025_min500_research.csv` | 639,744 | 479 | 2025-03-27 ~ 09-28 |
| `statcast_2026_min300_research.csv` | 391,038 | 452 | 2026-03-26 ~ 07-12 |

* 정규시즌(`game_type == R`)만, 투수-시즌 단위로 **시즌 투구 수 ≥ 500**(2026은 시즌 일부만 진행되어 **≥ 300**)인 투수의 모든 투구입니다.
* 파일명은 `statcast_<연도>_min<컷오프>_research.csv` 형식이어야 합니다. 형식이 어긋난 `statcast_*.csv`(예: 다운로드하며 붙은 ` (1)`)가 있으면 코드가 에러를 냅니다.
* 해외 개막전(2024 서울, 2025 도쿄 시리즈)과 2026-07-13 이후 경기는 CSV에 없습니다.
* 2026-10-01에 받은 파일은 Drive에서 6개 전부 `_research_score.csv`로(2026 파일은 `min500`으로) 올라와 있었습니다. 투수별 투구 수를 확인한 결과 2026 파일은 실제로는 기존과 같은 **300구 컷오프**(299~499구인 투수 99명 포함)였고, 6개 파일의 행·투수 수(3,592,302행, 1,067명)도 이전과 동일해 표본 자체는 바뀌지 않았습니다. 그래서 받은 뒤 이 폴더에는 기존 이름 규칙(`min500`/`min300`, `_research.csv`)으로 저장했습니다.

## 코드에서 쓰는 방식

CSV에는 2026-10-01부터 점수 관련 컬럼(`home_score`, `away_score`, `score_diff` 등)이 추가됐지만, 여전히 `fielder_2`(포수)는 없어서 CSV만으로는 분석 변수를 전부 만들 수 없습니다. 그래서 `data/raw`(수집한 전체 데이터)를 읽고 **CSV에 있는 행(`game_pk`, `at_bat_number`, `pitch_number`, `pitcher` 키 일치)만 남깁니다** (`src/preprocessing/research_sample.py`의 `load_research_pitches()`).

* 겹치는 열은 CSV와 raw가 같습니다. 예외로 `pitch_type` 12행만 다른데(Statcast 사후 재분류), 팀 표본과 똑같이 맞추려고 CSV 값을 씁니다.
* CSV의 행이 `data/raw`에 없으면 에러를 냅니다 (raw를 CSV 마지막 날짜까지 수집했는지 확인하세요).
* 폴더 위치는 기본 `data/research/`이고, `RESEARCH_SAMPLE_DIR` 환경변수로 바꿀 수 있습니다.
* 키 인덱스는 `data/processed/research_sample_index.parquet`에 캐시되며 CSV가 바뀌면 자동으로 다시 만듭니다.
