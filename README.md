# ⚾ 장타를 맞은 투수는 그 구종을 피하는가

**그리고 그 회피는 좋은 선택이었나** — YSAL 9기 야구 2팀 1차 프로젝트

MLB Statcast 2021 ~ 2026-07-12 정규시즌의 투구 3,592,302개(투수 1,067명)로, 투수가 장타를 맞은 구종을 같은 경기에서 그 타자를 다시 만났을 때 덜 던지는지, 그리고 그 회피가 좋은 선택이었는지를 분석했습니다.

📄 **전체 내용: [docs/최종정리.md](docs/최종정리.md)** · 수치 재현 결과: [docs/재현결과.md](docs/재현결과.md)

---

## 🔎 한눈에 보는 결론

| 질문 | 답 | 핵심 수치 |
|---|---|---|
| 장타 친 타자를 같은 경기에서 다시 만나면 맞은 구종을 덜 던지나? | **예** | 비슷한 상황의 아웃 뒤보다 **14.3%p** 덜 던짐, 다시 던질 오즈 **0.29배** |
| 경기 상황 때문 아닌가? | **아니다** | 주자·아웃·이닝·카운트·점수차를 맞춰도 14.3%p 그대로 |
| 누구를 향한 회피인가? | **그 타자** | 회피의 약 70%(9.5%p)가 장타를 친 그 타자에게 집중 |
| 어떤 장타일 때 더 피하나? | **홈런·강한 타구** | 홈런 여부 > 점수차 > 타구속도 순으로 영향, 볼카운트·주자는 작음 |
| 회피는 좋은 선택이었나? | **확정 못 함** | 모델 점수상 회피군이 약간 높지만(+0.0023) 득점 이득으로 볼 수 없고, 구사율별 유불리는 재검증에서 유지되지 않음 |

> **한 문장 결론.** 투수는 장타를 맞은 구종을 그 타자에게 확실히 덜 던진다. 그러나 현재 자료와 검증 결과만으로는 이 회피의 성과상 이점을 확정하기 어렵다.

---

## 🧭 분석 흐름

| 단계 | 내용 | 최종 정리 | 코드 |
|---|---|---|---|
| ① 회피 확인 | 순수 감소와 상황 맞추기 사다리(M0–M3), 타자 맞춤 회피, 다시 던질 때의 코스·구속 | p.3–5 | `src/report/avoidance.py` |
| ② 회피 요인 | 결과 × 타구 질(xwOBA·배럴), 상황 변수(GBM + SHAP) | p.6–7 | `src/report/factors.py` |
| ③ 회피 효과 평가 | GBM 선택점수, 맞은 구종 구사율과의 상호작용과 재검증 | p.8–10 | `src/report/effect.py` |
| ④ 사례·결론 | 회피율 상·하위 투수, 회피 성향의 시즌 안정성 | p.11 | `src/report/cases.py` |

p.7의 SHAP 분석은 이범석의 노트북(`SHAP_.ipynb`) 최종 모형을 옮긴 것입니다. 다른 단계와 달리 "같은 시즌 안의 다음 맞대결"을 재대결로 봅니다. 탐색 과정은 [Beomseok 브랜치의 GBM_analysis.md](https://github.com/Yonsei-Sports-Analytics-Lab/9th-first-project-baseball-2/blob/Beomseok/GBM_analysis.md)를 보세요.

---

## 💻 실행 방법

1. 저장소 클론 후 가상환경 생성: `python -m venv .venv` 후 활성화
2. 패키지 설치: `pip install -r requirements.txt`
3. 데이터 준비: [팀 공유 드라이브](https://drive.google.com/drive/folders/1nP5Z0f7q1cNftgKkVTsWe329ishi8iUM)의 분석 표본 CSV 6개를 받아 `data/research/`에 넣습니다 (파일 설명: [data/research/README.md](data/research/README.md)). 다른 폴더에 두었다면 `RESEARCH_SAMPLE_DIR` 환경변수로 지정합니다.
4. (①의 오즈비와 ③ 단계) R과 `lme4` 패키지가 필요합니다. 없으면 해당 항목은 건너뜁니다. p.7은 `shap` 패키지를 씁니다(`requirements.txt`에 포함).
5. 실행:

```bash
python main.py            # 전체 실행 (약 2.5시간, 대부분 p.10 부트스트랩)
python main.py --quick    # 약 4분: 부트스트랩 반복 수 축소, p.10 부트스트랩 생략
python main.py --steps 1 2
```

| 옵션 | 설명 |
|---|---|
| `--quick` | 부트스트랩·재학습 반복 수를 줄여 빠르게 확인합니다. 신뢰구간처럼 반복 수에 따라 달라지는 수치는 보고서와 다를 수 있습니다. |
| `--steps` | 실행할 단계(1~4)를 고릅니다. |
| `--boot N` | p.10 전체 과정 부트스트랩 반복 수를 직접 정합니다(전체 실행 기본 200). |

실행하면 `data/processed/final/`에 쪽별 표(CSV)와 그림이 저장되고, 보고서 수치와 다시 계산한 수치를 나란히 놓은 대조표(`report_check.md`)가 출력됩니다.

테스트: `python -m pytest`

---

## 📁 저장소 구조 (Repository Structure)

* `main.py`: 최종 정리의 수치를 순서대로 다시 계산하는 파이프라인
* `docs/최종정리.md`, `docs/figures/final/`: 최종 정리 문서와 그림
* `src/report/`: 파이프라인의 단계별 코드 (①~④)와 보고서 수치 대조
* `src/analysis/`: 분석 로직 (상황 매칭, 타자 맞춤 회피, 타구 질, 재대결 실행 비교, 선택점수, 혼합모형 실행)
* `src/preprocessing/`, `src/collection/`: 데이터 수집과 이벤트 데이터 구성
* `src/visualization/`: 파이프라인이 그리는 그림
* `tests/`: 단위 테스트
* `data/raw/`, `data/research/`, `data/processed/`: 원본 · 팀 공유 분석 표본 · 산출물 (모두 git에 올리지 않음)
* `notebooks/`: 탐색용 Jupyter Notebook (아래 브랜치 참고)

### 탐색 노트북과 상세 문서 (브랜치)

최종 정리에 쓰인 분석의 탐색 과정과 그 밖의 시도는 각자의 브랜치에 있습니다.

| 브랜치 | 내용 |
|---|---|
| [Chanyoung](https://github.com/Yonsei-Sports-Analytics-Lab/9th-first-project-baseball-2/tree/Chanyoung) | 노트북 03–07 (상황 매칭 사다리, 타구 질, 타자 맞춤 회피, 신뢰구간 점검), `docs/보고서.md` |
| [Dongyun (DongYun22 fork)](https://github.com/DongYun22/9th-first-project-baseball-2/tree/Dongyun) | 노트북 01–11 (대조군 설계, 재대결 실행 비교, RVAE·선택점수·구사율 상호작용) |
| [Beomseok](https://github.com/Yonsei-Sports-Analytics-Lab/9th-first-project-baseball-2/blob/Beomseok/GBM_analysis.md) | `GBM_analysis.md` (회피 요인 GBM + SHAP, 타구 질 분석) |

---

## ⚠️ 주의사항 (Precautions)

안전하고 원활한 협업을 위해 아래 사항을 반드시 숙지해 주세요.

* **대용량 데이터 업로드 금지:** 용량이 큰 CSV 파일이나 수만 건의 로우 데이터는 절대 GitHub에 직접 Commit/Push 하지 마세요. 반드시 `.gitignore`를 확인하고, 데이터는 구글 드라이브 등 별도의 스토리지를 통해 공유해야 합니다.
* **보안 정보 노출 주의:** API Key, 데이터베이스 비밀번호, 개인 정보 등은 절대 코드에 직접 작성하지 마세요. `.env` 파일을 활용하고 해당 파일이 `.gitignore`에 포함되어 있는지 확인해야 합니다.
* **Main 브랜치 직접 Push 금지:** `main` 브랜치에 코드를 직접 올리는 것은 금지되어 있습니다. 반드시 각자의 작업 브랜치를 생성(`feat/데이터수집` 등)하여 작업한 후, Pull Request(PR)를 통해 코드 리뷰를 거쳐 병합(Merge)하세요.
* **환경 동기화:** 새로운 라이브러리를 설치한 경우, 반드시 `requirements.txt`를 업데이트하고 커밋해 주세요.
* **이슈 트래킹:** 새로운 작업을 시작하거나 버그를 발견했을 때는 항상 템플릿에 맞추어 `Issue`를 먼저 등록해 주세요.

---

## 👥 팀원

곽동윤 · 김민정 · 김찬영 · 이범석 · 이인규
