"""Numbers printed in docs/최종정리.md, keyed the same way the pipeline
steps report theirs, so `main.py` can lay the two side by side.

Each entry: key -> (page, label, report value, decimals). A value is
compared after rounding both sides to `decimals`; text values are compared
as they are. `BOOTSTRAP_KEYS` depend on the number of bootstrap/refit
repetitions, so they can only match in a full run.
"""

from __future__ import annotations

import pandas as pd

REPORT_VALUES = {
    # p.2-3: avoidance in the rematch
    "n_xbh_total": (2, "장타 건수", 70_319, 0),
    "n_rematch": (2, "같은 타자 재대결 건수", 27_391, 0),
    "ladder_M0_net": (3, "순수 감소 M0 (%p)", 14.36, 2),
    "ladder_M1_net": (3, "순수 감소 M1 (%p)", 14.36, 2),
    "ladder_M2_net": (3, "순수 감소 M2 (%p)", 14.32, 2),
    "ladder_M3_net": (3, "순수 감소 M3 (%p)", 14.31, 2),
    "ladder_M3_ci_low": (3, "순수 감소 M3 95% CI 하한 (%p)", 13.8, 1),
    "ladder_M3_ci_high": (3, "순수 감소 M3 95% CI 상한 (%p)", 14.8, 1),
    "next_batter_M3_net": (3, "바로 다음 타자 순수 감소 M3 (%p)", 6.75, 2),
    "runners_smd_M0": (3, "M0 주자 수 SMD", 0.12, 2),
    "odds_M0": (3, "다시 던질 오즈비 M0", 0.30, 2),
    "odds_M1": (3, "다시 던질 오즈비 M1", 0.28, 2),
    "odds_M2": (3, "다시 던질 오즈비 M2", 0.29, 2),
    "odds_M3": (3, "다시 던질 오즈비 M3", 0.29, 2),
    "odds_M3_ci_low": (3, "오즈비 M3 95% CI 하한", 0.27, 2),
    "odds_M3_ci_high": (3, "오즈비 M3 95% CI 상한", 0.31, 2),
    # p.4: aimed at that batter
    "avoid_inter": (4, "중간 타자 추가 회피 (%p)", 4.0, 1),
    "avoid_inter_ci_low": (4, "중간 타자 95% CI 하한", 3.8, 1),
    "avoid_inter_ci_high": (4, "중간 타자 95% CI 상한", 4.2, 1),
    "avoid_rematch": (4, "재대결 추가 회피 (%p)", 13.5, 1),
    "avoid_rematch_ci_low": (4, "재대결 95% CI 하한", 13.0, 1),
    "avoid_rematch_ci_high": (4, "재대결 95% CI 상한", 14.0, 1),
    "batter_specific": (4, "타자 맞춤 회피 (%p)", 9.5, 1),
    "batter_specific_ci_low": (4, "타자 맞춤 95% CI 하한", 9.0, 1),
    "batter_specific_ci_high": (4, "타자 맞춤 95% CI 상한", 10.0, 1),
    "batter_specific_share": (4, "타자 맞춤 비중 (%)", 70, 0),
    "avoid_prior": (4, "장타 전 그 타자 회피 (%p, 음수 = 더 던짐)", -1.5, 1),
    # p.4-5: how the type is thrown when reused
    "reuse_share_xbh": (4, "다시 던진 비율: 장타 (%)", 45.7, 1),
    "reuse_share_control": (4, "다시 던진 비율: 대조군 (%)", 67.7, 1),
    "plate_z_cm_R_SL": (4, "우투 슬라이더 높낮이 차이 (cm)", -4, 0),
    "plate_z_cm_R_ST": (4, "우투 스위퍼 높낮이 차이 (cm)", -4, 0),
    "plate_z_cm_R_CH": (4, "우투 체인지업 높낮이 차이 (cm)", -4, 0),
    "plate_z_cm_L_CU": (4, "좌투 커브 높낮이 차이 (cm)", -7, 0),
    "n_comparisons": (5, "구종×지표 비교 수", 112, 0),
    "n_distinct": (5, "0과 구별된 비교 수", 16, 0),
    # p.6: result and contact quality
    "hq_아웃_Q4": (6, "아웃, 타구 질 상위 25% (%p)", 8.6, 1),
    "hq_단타_Q1": (6, "단타, 하위 25% (%p)", -0.7, 1),
    "hq_단타_Q4": (6, "단타, 상위 25% (%p)", 10.9, 1),
    "hq_2·3루타_Q1": (6, "2·3루타, 하위 25% (%p)", 3.7, 1),
    "hq_2·3루타_Q4": (6, "2·3루타, 상위 25% (%p)", 15.0, 1),
    "hq_홈런_Q1": (6, "홈런, 하위 25% (%p)", 16.9, 1),
    "hq_홈런_Q4": (6, "홈런, 상위 25% (%p)", 19.0, 1),
    "barrel_out": (6, "배럴 아웃 − 비배럴 아웃 (%p)", 10.1, 1),
    "n_weak_home_run": (6, "약한 홈런 건수", 37, 0),
    # p.7: situation factors (GBM + SHAP)
    "shap_n": (7, "SHAP 분석 이벤트 수 (같은 시즌 내 재대결)", 38_012, 0),
    "shap_r2": (7, "회피 요인 GBM 테스트 R²", 0.035, 3),
    "shap_top3": (7, "평균 |SHAP| 상위 3개 변수", "홈런 여부 > 점수차 > 타구 속도", None),
    # p.8-9: selection score
    "gbm_validation_r2": (8, "GBM 검증 R² (2021–25 학습 → 2026)", 0.057, 3),
    "selection_n": (9, "선택점수 재대결 건수", 27_245, 0),
    "selection_n_pitchers": (9, "선택점수 투수 수", 695, 0),
    "selection_gap": (9, "회피군 − 재사용군 선택점수", 0.0023, 4),
    "selection_ci_low": (9, "보수적 95% CI 하한", 0.0009, 4),
    "selection_ci_high": (9, "보수적 95% CI 상한", 0.0037, 4),
    "selection_p": (9, "보수적 p값", 0.002, 3),
    # p.10: usage interaction
    "interaction_base_n": (10, "기존 사양 건수", 10_747, 0),
    "interaction_base": (10, "기존 사양 회피 × 구사율 계수", -0.0645, 4),
    "interaction_base_ci_low": (10, "기존 사양 95% 구간 하한 (GBM 고정)", -0.0749, 4),
    "interaction_base_ci_high": (10, "기존 사양 95% 구간 상한 (GBM 고정)", -0.0541, 4),
    "interaction_pooled_n": (10, "경기 전 구사율·시간 외 합산 건수", 8_813, 0),
    "interaction_pooled": (10, "경기 전 구사율·시간 외 합산 계수", 0.0061, 4),
    "interaction_pooled_ci_low": (10, "시간 외 합산 부트스트랩 구간 하한", -0.0352, 4),
    "interaction_pooled_ci_high": (10, "시간 외 합산 부트스트랩 구간 상한", 0.0281, 4),
    "interaction_primary_n": (10, "2026 평가 건수", 1_096, 0),
    "interaction_primary": (10, "2021–25 학습 → 2026 평가 계수", -0.0283, 4),
    "interaction_primary_ci_low": (10, "2026 평가 부트스트랩 구간 하한", -0.1061, 4),
    "interaction_primary_ci_high": (10, "2026 평가 부트스트랩 구간 상한", 0.0113, 4),
    "interaction_no_usage": (10, "GBM 구사율 입력 제거 시 계수", 0.0127, 4),
    # p.11: pitcher cases
    "cases_top": (11, "회피 상위 5명", "Schultz 92% · Arihara 91% · Schwellenbach 83% · Crismatt 82% · Brieske 81%", None),
    "cases_bottom": (11, "회피 하위 5명", "Yesavage 10% · Widener 14% · McKenzie 15% · Abbott 17% · Sasaki 18%", None),
    "omega_squared": (11, "회피 성향의 투수 고유 몫 (ω²)", 0.27, 2),
}

BOOTSTRAP_KEYS = {
    "ladder_M3_ci_low", "ladder_M3_ci_high", "selection_ci_low", "selection_ci_high", "selection_p",
    "interaction_pooled_ci_low", "interaction_pooled_ci_high", "interaction_primary_ci_low", "interaction_primary_ci_high",
}


def _show(value, decimals) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if decimals is None:
        return str(value)
    return f"{value:,.{decimals}f}"


def compare(results: dict, full_run: bool) -> pd.DataFrame:
    """One row per report number: the report's value, the reproduced value and whether they agree."""
    rows = []
    for key, (page, label, expected, decimals) in REPORT_VALUES.items():
        actual = results.get(key)
        if actual is None or (isinstance(actual, float) and pd.isna(actual)):
            status = "건너뜀"
        elif decimals is None:
            status = "일치" if str(actual) == str(expected) else "차이"
        else:
            status = "일치" if round(float(actual), decimals) == round(float(expected), decimals) else "차이"
        if status == "차이" and key in BOOTSTRAP_KEYS and not full_run:
            status = "차이 (반복 수 축소)"
        rows.append({"쪽": page, "항목": label, "보고서": _show(expected, decimals), "재현": _show(actual, decimals),
                     "결과": status})
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    """The comparison table as a GitHub-flavoured markdown table."""
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *body])
