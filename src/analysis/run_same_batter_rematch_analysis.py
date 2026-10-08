"""Pitch-avoidance analysis restricted to SAME-BATTER REMATCHES (main analysis).

The next-batter analysis (run_pitch_avoidance_analysis.py) compares against
the very next batter (a different batter). This script instead keeps only
extra-base hits where the same pitcher faces the SAME batter again in that
batter's immediately following plate appearance of the game, and asks
whether the pitcher avoids the pitch type that batter just hit.

The sample is the team's research sample (data/raw restricted to the rows in
the research CSVs, see src/preprocessing/research_sample.py).

Everything is built with the exact same event builder as the other analysis
(build_event_dataset_for_events with next_pa_mode="same_batter"), and the
placebo group (field_out) is drawn only from field_outs that also have a
same-batter rematch, stratified to the XBH group's season x pitch_family
distribution.

Sections:
1. XBH-only paired drop against two baselines (within-game baseline_usage and
   the pitcher's season-wide usage rate).
2. Placebo-netted diff-in-diff (overall / pitch family / season / baseline>0).
3. Binary reuse comparison.
4. Mixed-effects logistic model (needs R; skipped if absent).
5. Platoon (same- vs opposite-handed): mixed model with platoon_match and
   group:platoon_match, plus net-effect breakdowns by handedness and by
   specific pitch type.
6. Same-pitcher matching robustness: control drawn only from the same
   pitcher x season x pitch_family strata as the XBH events (both groups
   trimmed to identical stratum counts); compared against section 4.
"""

import logging
from pathlib import Path

import pandas as pd
from scipy import stats

from src.analysis.avoidance_stats import (
    build_stratified_placebo_candidates,
    compute_expected_reuse_prob,
    compute_season_usage_rate,
    match_treatment_to_control,
    summarize_diff_in_diff,
    summarize_paired_diff,
)
from src.analysis.mixed_effects_model import (
    MODEL_FORMULA_PLATOON,
    MODEL_FORMULA_ROBUSTNESS,
    REQUIRED_MODEL_COLUMNS,
    build_combined_model_dataset,
    compare_random_effects,
)
from src.preprocessing.build_next_ab_dataset import (
    build_event_dataset_for_events,
    filter_to_same_batter_rematch,
    identify_extra_base_hit_events,
)
from src.preprocessing.research_sample import load_research_pitches

try:
    from src.analysis.glmer_runner import (
        check_convergence,
        extract_fixed_effects_table,
        extract_pitcher_variance_components,
        extract_random_effects_table,
        fit_glmer,
    )

    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
XBH_REMATCH_PATH = PROCESSED_DATA_DIR / "pitch_reuse_after_xbh_same_batter_rematch_events.parquet"
PLACEBO_REMATCH_PATH = PROCESSED_DATA_DIR / "pitch_reuse_placebo_same_batter_rematch_events.parquet"
PLACEBO_PITCHER_MATCHED_PATH = PROCESSED_DATA_DIR / "pitch_reuse_placebo_same_batter_rematch_pitcher_matched_events.parquet"
RANDOM_EFFECTS_PATH = PROCESSED_DATA_DIR / "pitch_avoidance_same_batter_rematch_random_effects.csv"
PLATOON_FIXED_EFFECTS_PATH = PROCESSED_DATA_DIR / "pitch_avoidance_same_batter_rematch_platoon_fixed_effects.csv"
PITCHER_MATCHED_FIXED_EFFECTS_PATH = PROCESSED_DATA_DIR / "pitch_avoidance_same_batter_rematch_pitcher_matched_fixed_effects.csv"
PLACEBO_OUTCOME_EVENTS = {"field_out"}
PLACEBO_SEED = 20260923
DEFAULT_STRATA = ("season", "pitch_family")
PITCHER_STRATA = ("pitcher", "season", "pitch_family")
MIN_BREAKDOWN_N = 300
NEEDED_COLUMNS = [
    "game_pk", "game_date", "season", "pitcher", "player_name", "batter", "pitch_type", "events",
    "at_bat_number", "pitch_number", "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "stand", "p_throws", "home_score", "away_score", "fielder_2",
]
KEY_TERMS = ["group", "group:baseline_usage", "platoon_match", "group:platoon_match"]


def load_pitches(extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    return load_research_pitches()[NEEDED_COLUMNS + list(extra_columns)].copy()


def attach_season_usage(events: pd.DataFrame, season_usage: pd.DataFrame) -> pd.DataFrame:
    return events.merge(
        season_usage[["pitcher", "season", "pitch_type", "season_usage_rate"]],
        left_on=["pitcher", "season", "hit_pitch_type"],
        right_on=["pitcher", "season", "pitch_type"],
        how="left",
    ).drop(columns=["pitch_type"])


def build_xbh_rematch(pitches: pd.DataFrame, season_usage: pd.DataFrame) -> pd.DataFrame:
    xbh_all = identify_extra_base_hit_events(pitches)
    candidates = filter_to_same_batter_rematch(pitches, xbh_all)
    by_season = pd.DataFrame({
        "xbh_total": xbh_all.groupby("season").size(),
        "same_batter_rematch": candidates.groupby("season").size(),
    })
    by_season["rematch_share"] = by_season["same_batter_rematch"] / by_season["xbh_total"]
    logger.info(
        "장타 %d건 중 같은 타자·같은 투수 재대결 %d건 (%.1f%%)\n%s",
        len(xbh_all), len(candidates), len(candidates) / len(xbh_all) * 100, by_season.round(4).to_string(),
    )
    xbh = build_event_dataset_for_events(pitches, candidates, next_pa_mode="same_batter")
    return attach_season_usage(xbh, season_usage)


def build_placebo_rematch(
    pitches: pd.DataFrame,
    xbh: pd.DataFrame,
    season_usage: pd.DataFrame,
    strata_cols: tuple[str, ...] = DEFAULT_STRATA,
) -> pd.DataFrame:
    candidates = build_stratified_placebo_candidates(
        pitches, xbh, PLACEBO_OUTCOME_EVENTS, seed=PLACEBO_SEED,
        candidate_filter=filter_to_same_batter_rematch, strata_cols=strata_cols,
    )
    placebo = build_event_dataset_for_events(pitches, candidates, next_pa_mode="same_batter")
    return attach_season_usage(placebo, season_usage)


def _usable(events: pd.DataFrame, baseline_col: str) -> pd.DataFrame:
    d = events[events["has_next_ab"] & events[baseline_col].notna()].copy()
    d["diff"] = d[baseline_col] - d["same_type_share"]
    return d


def report_paired(label: str, events: pd.DataFrame, baseline_col: str) -> None:
    d = events[events["has_next_ab"] & events[baseline_col].notna()]
    r = summarize_paired_diff(d[baseline_col] - d["same_type_share"])
    logger.info(
        "[%s] 사전 %.4f -> 사후 %.4f, 감소 %.4f (95%% CI [%.4f, %.4f]), n=%d, p=%.2e",
        label, d[baseline_col].mean(), d["same_type_share"].mean(), r["mean"], r["ci_low"], r["ci_high"], r["n"], r["t_p"],
    )


def report_season_table(events: pd.DataFrame, baseline_col: str) -> None:
    d = events[events["has_next_ab"] & events[baseline_col].notna()]
    rows = []
    for season, g in d.groupby("season"):
        rows.append({
            "season": season, "events": len(g), "baseline": g[baseline_col].mean(),
            "post": g["same_type_share"].mean(), "diff(post-baseline)": g["same_type_share"].mean() - g[baseline_col].mean(),
        })
    logger.info("[XBH 재대결, %s 기준] 시즌별\n%s", baseline_col, pd.DataFrame(rows).round(4).to_string(index=False))


def report_diff_in_diff(label: str, xbh: pd.DataFrame, placebo: pd.DataFrame, baseline_col: str, detail: bool = True) -> None:
    x, p = _usable(xbh, baseline_col), _usable(placebo, baseline_col)
    o = summarize_diff_in_diff(x["diff"], p["diff"])
    logger.info(
        "[%s / %s] n_XBH=%d n_placebo=%d | XBH 감소 %.4f, placebo 감소 %.4f, 순수효과 %.4f (95%% CI [%.4f, %.4f]) Welch p=%.2e MW p=%.2e",
        label, baseline_col, o["n_treatment"], o["n_control"], o["treatment_mean"], o["control_mean"],
        o["net_effect"], o["ci_low"], o["ci_high"], o["t_p"], o["u_p"],
    )
    if not detail:
        return
    for fam in ["fastball", "breaking", "offspeed"]:
        r = summarize_diff_in_diff(x[x["pitch_family"] == fam]["diff"], p[p["pitch_family"] == fam]["diff"])
        logger.info(
            "    family=%-8s n=%d/%d XBH %.4f placebo %.4f 순수 %.4f [%.4f, %.4f]",
            fam, r["n_treatment"], r["n_control"], r["treatment_mean"], r["control_mean"], r["net_effect"], r["ci_low"], r["ci_high"],
        )
    for season in sorted(x["season"].unique()):
        r = summarize_diff_in_diff(x[x["season"] == season]["diff"], p[p["season"] == season]["diff"])
        logger.info(
            "    season=%s n=%d/%d XBH %.4f placebo %.4f 순수 %.4f [%.4f, %.4f]",
            season, r["n_treatment"], r["n_control"], r["treatment_mean"], r["control_mean"], r["net_effect"], r["ci_low"], r["ci_high"],
        )
    r = summarize_diff_in_diff(x[x[baseline_col] > 0]["diff"], p[p[baseline_col] > 0]["diff"])
    logger.info(
        "    robustness(baseline>0) n=%d/%d XBH %.4f placebo %.4f 순수 %.4f [%.4f, %.4f]",
        r["n_treatment"], r["n_control"], r["treatment_mean"], r["control_mean"], r["net_effect"], r["ci_low"], r["ci_high"],
    )


def report_breakdown(label: str, xbh: pd.DataFrame, placebo: pd.DataFrame, baseline_col: str, by: str) -> None:
    """Net (placebo-netted) drop for each level of `by`, keeping levels with at
    least MIN_BREAKDOWN_N usable events in BOTH groups.
    """
    x, p = _usable(xbh, baseline_col), _usable(placebo, baseline_col)
    rows = []
    for level in sorted(set(x[by].dropna().unique()) & set(p[by].dropna().unique())):
        gx, gp = x[x[by] == level]["diff"], p[p[by] == level]["diff"]
        if len(gx) < MIN_BREAKDOWN_N or len(gp) < MIN_BREAKDOWN_N:
            continue
        r = summarize_diff_in_diff(gx, gp)
        rows.append({
            by: level, "n_xbh": r["n_treatment"], "n_placebo": r["n_control"], "xbh_drop": r["treatment_mean"],
            "placebo_drop": r["control_mean"], "net": r["net_effect"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
        })
    logger.info("[%s / %s 기준] %s별 순수효과 (양쪽 표본 >=%d)\n%s", label, baseline_col, by, MIN_BREAKDOWN_N, pd.DataFrame(rows).round(4).to_string(index=False))


def report_binary(xbh: pd.DataFrame, placebo: pd.DataFrame, baseline_col: str) -> None:
    def prep(events: pd.DataFrame) -> pd.DataFrame:
        d = events[events["has_next_ab"] & events[baseline_col].notna()].copy()
        d["reuse_gap"] = d["reused_same_type"] - compute_expected_reuse_prob(d[baseline_col], d["next_ab_pitch_count"])
        return d

    x, p = prep(xbh), prep(placebo)
    g = summarize_diff_in_diff(x["reuse_gap"], p["reuse_gap"], alternative="less")
    contingency = pd.DataFrame(
        {
            "reused=1": [x["reused_same_type"].sum(), p["reused_same_type"].sum()],
            "reused=0": [(x["reused_same_type"] == 0).sum(), (p["reused_same_type"] == 0).sum()],
        },
        index=["XBH", "placebo"],
    )
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    logger.info(
        "[이진 재사용 / %s] raw 재사용률 XBH %.4f vs placebo %.4f (chi2=%.1f p=%.2e) | 기대 대비 gap XBH %.4f placebo %.4f 순수 %.4f (p=%.2e)",
        baseline_col, x["reused_same_type"].mean(), p["reused_same_type"].mean(), chi2, chi_p,
        g["treatment_mean"], g["control_mean"], g["net_effect"], g["t_p"],
    )


def fit_mixed(label: str, xbh: pd.DataFrame, placebo: pd.DataFrame, formula: str, extra_required: tuple[str, ...] = ()) -> dict:
    xbh_f = xbh[xbh["has_next_ab"] & xbh["baseline_usage"].notna()]
    placebo_f = placebo[placebo["has_next_ab"] & placebo["baseline_usage"].notna()]
    combined = build_combined_model_dataset(
        xbh_f, placebo_f, required_columns=REQUIRED_MODEL_COLUMNS + list(extra_required)
    )
    n_xbh, n_placebo = int((combined["group"] == 1).sum()), int((combined["group"] == 0).sum())
    logger.info(
        "[혼합효과 / %s] 표본 %d행 (XBH %d + placebo %d), 투수 %d명\n  공식: %s",
        label, len(combined), n_xbh, n_placebo, combined["pitcher"].nunique(), formula,
    )
    fit_glmer(combined, formula)
    convergence = check_convergence()
    fe = extract_fixed_effects_table()
    _, group_var, _ = extract_pitcher_variance_components()
    logger.info("[혼합효과 / %s] 수렴: %s | 투수 group 기울기 분산=%.4f", label, convergence, group_var)
    return {
        "label": label, "n_rows": len(combined), "n_xbh": n_xbh, "n_placebo": n_placebo,
        "n_pitchers": combined["pitcher"].nunique(), "convergence": convergence, "fe": fe,
        "group_var": group_var, "ranef": extract_random_effects_table(),
    }


def log_key_terms(result: dict, terms: list[str]) -> None:
    fe = result["fe"]
    view = fe[fe["term"].isin(terms)][["term", "estimate", "std_error", "p_value", "odds_ratio", "or_ci_low", "or_ci_high"]]
    logger.info("[혼합효과 / %s] 핵심 계수 (오즈비, Wald 95%% CI)\n%s", result["label"], view.round(4).to_string(index=False))


def run() -> None:
    pitches = load_pitches()
    season_usage = compute_season_usage_rate(pitches)

    xbh = build_xbh_rematch(pitches, season_usage)
    placebo = build_placebo_rematch(pitches, xbh, season_usage)
    logger.info("placebo(field_out 재대결) %d건 층화추출 (XBH 재대결 %d건에 맞춤)", len(placebo), len(xbh))
    xbh.to_parquet(XBH_REMATCH_PATH, index=False)
    placebo.to_parquet(PLACEBO_REMATCH_PATH, index=False)

    logger.info("=== 1. XBH 재대결만 (대조군 없음) ===")
    report_paired("경기 내 baseline", xbh, "baseline_usage")
    report_paired("시즌 baseline", xbh, "season_usage_rate")
    report_season_table(xbh, "season_usage_rate")

    logger.info("=== 2. placebo(field_out 재대결) 넷팅 diff-in-diff ===")
    report_diff_in_diff("재대결", xbh, placebo, "season_usage_rate")
    report_diff_in_diff("재대결", xbh, placebo, "baseline_usage")

    logger.info("=== 3. 이진 재사용 지표 ===")
    report_binary(xbh, placebo, "season_usage_rate")

    if not R_AVAILABLE:
        logger.warning("rpy2/R 없음 -- 혼합효과 모델(4~6절) 생략")
        return

    logger.info("=== 4. 혼합효과 로지스틱 회귀 (robustness 공식) ===")
    main = fit_mixed("메인(층화 대조군)", xbh, placebo, MODEL_FORMULA_ROBUSTNESS)
    log_key_terms(main, KEY_TERMS[:2])
    main["ranef"].to_csv(RANDOM_EFFECTS_PATH, index=False)
    logger.info("투수별 랜덤효과 저장: %s", RANDOM_EFFECTS_PATH)

    logger.info("=== 5. Platoon (동타=1 / 이타=0) ===")
    platoon = fit_mixed("platoon 모델", xbh, placebo, MODEL_FORMULA_PLATOON, extra_required=("platoon_match",))
    log_key_terms(platoon, KEY_TERMS)
    platoon["fe"].assign(model=platoon["label"]).to_csv(PLATOON_FIXED_EFFECTS_PATH, index=False)
    for by in ["platoon_match", "stand", "p_throws"]:
        report_breakdown("재대결", xbh, placebo, "season_usage_rate", by)
    report_breakdown("재대결", xbh, placebo, "season_usage_rate", "hit_pitch_type")

    logger.info("=== 6. 투수 단위 매칭 robustness (같은 투수 x 시즌 x 구종계열의 field_out만 대조군) ===")
    placebo_pm = build_placebo_rematch(pitches, xbh, season_usage, strata_cols=PITCHER_STRATA)
    xbh_pm = match_treatment_to_control(xbh, placebo_pm, PITCHER_STRATA, seed=PLACEBO_SEED)
    placebo_pm.to_parquet(PLACEBO_PITCHER_MATCHED_PATH, index=False)
    logger.info(
        "같은 투수 대조군 %d건 (장타 %d건 중 %d건이 짝을 찾음 = %.1f%%), 투수 %d명",
        len(placebo_pm), len(xbh), len(xbh_pm), len(xbh_pm) / len(xbh) * 100, xbh_pm["pitcher"].nunique(),
    )
    report_diff_in_diff("투수매칭", xbh_pm, placebo_pm, "season_usage_rate", detail=False)
    report_diff_in_diff("투수매칭", xbh_pm, placebo_pm, "baseline_usage", detail=False)
    matched = fit_mixed("투수매칭", xbh_pm, placebo_pm, MODEL_FORMULA_ROBUSTNESS)
    log_key_terms(matched, KEY_TERMS[:2])
    both = main["fe"][["term", "estimate", "odds_ratio", "p_value"]].merge(
        matched["fe"][["term", "estimate", "odds_ratio", "p_value"]], on="term", suffixes=("_층화", "_투수매칭")
    )
    logger.info("[고정효과 비교: 층화 대조군 vs 투수매칭]\n%s", both.round(4).to_string(index=False))
    corr = compare_random_effects(main["ranef"], matched["ranef"])
    logger.info(
        "[투수별 랜덤 기울기 상관, 공통 투수 %d명] Pearson=%.4f Spearman=%.4f",
        corr["n_common"], corr["pearson"], corr["spearman"],
    )
    matched["fe"].assign(model=matched["label"]).to_csv(PITCHER_MATCHED_FIXED_EFFECTS_PATH, index=False)


if __name__ == "__main__":
    run()
