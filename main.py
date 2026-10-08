"""Final-report pipeline: reproduces the numbers in docs/최종정리.md.

    python main.py            # full run (about 2.5 hours; nearly all of it is the p.10 bootstrap)
    python main.py --quick    # fewer bootstrap repetitions, no p.10 bootstrap (about 4 minutes)
    python main.py --steps 1 2

Steps follow the report: 1 회피 확인 (p.3-5) -> 2 회피 요인 (p.6-7) -> 3 회피 효과 평가
(p.8-10) -> 4 사례 (p.11). Tables and figures go to data/processed/final/, and a table
comparing each reproduced number with the report's is printed and saved as report_check.md.

Data: put the research-sample CSVs in data/research/ (see data/research/README.md) or point
RESEARCH_SAMPLE_DIR at them. The odds ratios (step 1) and step 3 need R with lme4 (via rpy2);
p.7 needs the shap package.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import pandas as pd

from src.analysis.avoidance_stats import compute_season_usage_rate
from src.analysis.run_same_batter_rematch_analysis import build_xbh_rematch
from src.preprocessing.build_next_ab_dataset import identify_extra_base_hit_events
from src.report import avoidance, cases, factors
from src.report.data import load_pitches
from src.report.reference import compare, to_markdown
from src.visualization import final_report_plots as plots

OUT_DIR = Path(__file__).resolve().parent / "data" / "processed" / "final"
FIG_DIR = OUT_DIR / "figures"
FT_TO_CM = 30.48
# repetitions: ladder cluster bootstrap, GBM refits and permutations (p.9), full-pipeline bootstrap (p.10)
FULL = {"ladder_boot": 1000, "refit": 50, "perm": 5000, "interaction_boot": 200}
QUICK = {"ladder_boot": 200, "refit": 5, "perm": 500, "interaction_boot": 0}

logger = logging.getLogger("final_report")


def r_available() -> bool:
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(lme4))")
        return True
    except Exception as exc:  # R, rpy2 or lme4 missing
        logger.warning("R/lme4를 쓸 수 없어 혼합모형 단계를 건너뜁니다: %s", str(exc)[:200])
        return False


def save(table: pd.DataFrame, name: str) -> None:
    table.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def step1_avoidance(pitches, season_usage, xbh, budget, use_r) -> dict:
    results = {"n_xbh_total": len(identify_extra_base_hit_events(pitches)), "n_rematch": len(xbh)}

    ladder = avoidance.weighted_ladder(pitches, season_usage, budget["ladder_boot"])
    save(ladder, "p03_ladder_weighted.csv")
    plots.plot_ladder(ladder, FIG_DIR / "p03_ladder.png")
    rematch = ladder[ladder["mode"] == "same_batter"].set_index("step")
    for step in rematch.index:
        results[f"ladder_{step}_net"] = rematch.loc[step, "net"] * 100
    results["ladder_M3_ci_low"], results["ladder_M3_ci_high"] = rematch.loc["M3", ["ci_low", "ci_high"]] * 100
    results["next_batter_M3_net"] = ladder[(ladder["mode"] == "next_batter") & (ladder["step"] == "M3")]["net"].iloc[0] * 100

    balance, pairs = avoidance.matched_ladder(pitches, season_usage)
    save(balance, "p03_ladder_matched_balance.csv")
    results["runners_smd_M0"] = balance.set_index("step").loc["M0", "runners_smd"]
    if use_r:
        odds = avoidance.ladder_odds_ratios(pairs)
        save(odds, "p03_ladder_odds_ratio.csv")
        for _, row in odds.iterrows():
            results[f"odds_{row['step']}"] = row["odds_ratio"]
        m3 = odds.set_index("step").loc["M3"]
        results["odds_M3_ci_low"], results["odds_M3_ci_high"] = m3["or_ci_low"], m3["or_ci_high"]

    xbh_m3, control_m3 = pairs["M3"]
    summary, by_slot, prior = avoidance.batter_specific_tables(pitches, xbh_m3, control_m3)
    save(summary, "p04_batter_specific.csv")
    save(by_slot, "p04_batter_specific_by_slot.csv")
    save(prior, "p04_batter_specific_prior.csv")
    plots.plot_slots(by_slot, FIG_DIR / "p04_batter_specific_by_slot.png")
    s = summary.set_index("measure") * 100
    for measure in ("avoid_inter", "avoid_rematch", "batter_specific"):
        results[measure] = s.loc[measure, "diff"]
        results[f"{measure}_ci_low"], results[f"{measure}_ci_high"] = s.loc[measure, "ci_low"], s.loc[measure, "ci_high"]
    results["batter_specific_share"] = s.loc["batter_specific", "diff"] / s.loc["avoid_rematch", "diff"] * 100
    results["avoid_prior"] = prior.set_index("measure").loc["avoid_prior", "diff"] * 100

    reuse, differences = avoidance.rematch_execution_tables(pitches, xbh_m3, control_m3)
    save(reuse, "p04_reuse_share.csv")
    save(differences, "p05_rematch_vs_control.csv")
    for hand, name in (("R", "RHP"), ("L", "LHP")):
        plots.plot_rematch_vs_control(differences, hand, FIG_DIR / f"p05_rematch_vs_control_{name}.png")
    shares = reuse.set_index("group")["reuse_share"] * 100
    results["reuse_share_xbh"], results["reuse_share_control"] = shares["xbh"], shares["control"]
    height = differences[differences["measure"] == "plate_z"].set_index(["p_throws", "hit_pitch_type"])["diff"]
    for hand, pitch_type in (("R", "SL"), ("R", "ST"), ("R", "CH"), ("L", "CU")):
        if (hand, pitch_type) in height.index:
            results[f"plate_z_cm_{hand}_{pitch_type}"] = height.loc[(hand, pitch_type)] * FT_TO_CM
    results["n_comparisons"], results["n_distinct"] = len(differences), int(differences["distinct"].sum())
    return results


def step2_factors(pitches, season_usage) -> dict:
    cells, barrel = factors.hit_quality_tables(pitches, season_usage)
    save(cells, "p06_hit_quality_xwoba.csv")
    save(barrel, "p06_hit_quality_barrel.csv")
    plots.plot_hit_quality(cells, FIG_DIR / "p06_hit_quality.png")
    by_cell = cells.set_index(["outcome", "xwoba_q"])
    results = {f"hq_{outcome}_{q}": by_cell.loc[(outcome, q), "vs_ref"] * 100
               for outcome in ("아웃", "단타", "2·3루타", "홈런") for q in ("Q1", "Q4") if (outcome, q) != ("아웃", "Q1")}
    results["barrel_out"] = barrel.set_index(["outcome", "barrel_label"]).loc[("아웃", "배럴"), "vs_ref"] * 100
    results["n_weak_home_run"] = int(by_cell.loc[("홈런", "Q1"), "n"])
    try:
        import shap  # noqa: F401
    except ImportError:
        logger.warning("shap 패키지가 없어 p.7 SHAP 분석을 건너뜁니다 (pip install shap)")
        return results
    shap_result = factors.shap_factor_tables(pitches, season_usage)
    importance = shap_result["importance"]
    save(importance, "p07_shap_importance.csv")
    plots.plot_shap_importance(importance, FIG_DIR / "p07_shap_importance.png")
    results.update({"shap_n": shap_result["n"], "shap_r2": shap_result["r2"],
                    "shap_top3": " > ".join(importance["label"].head(3))})
    return results


def step3_effect(pitches, season_usage, xbh, budget) -> dict:
    from src.report import effect  # imports rpy2

    ctx = effect.prepare(pitches, season_usage, xbh)
    results = {"gbm_validation_r2": ctx["validation_r2"]}

    gap = effect.selection_score_gap(ctx, budget["refit"], budget["perm"])
    save(pd.DataFrame([gap]), "p09_selection_score.csv")
    results.update({"selection_n": gap["n"], "selection_n_pitchers": gap["n_pitchers"], "selection_gap": gap["estimate"],
                    "selection_ci_low": gap.get("ci_low"), "selection_ci_high": gap.get("ci_high"),
                    "selection_p": gap.get("p_value")})

    interaction = effect.usage_interaction(ctx, pitches, budget["interaction_boot"])
    table = interaction["table"]
    save(table, "p10_usage_interaction.csv")
    save(interaction["curve"], "p10_delta_by_usage.csv")
    if len(interaction["bootstrap"]):
        save(interaction["bootstrap"], "p10_bootstrap_draws.csv")
    plots.plot_delta_by_usage(interaction["curve"], interaction["usage_by_group"], FIG_DIR / "p10_delta_by_usage.png")
    for name, (_, row) in zip(("base", "pooled", "primary"), table.iterrows()):
        results[f"interaction_{name}_n"] = row["n"]
        results[f"interaction_{name}"] = row["interaction"]
        if name == "base" or row["ci_kind"] != "GBM 고정":  # the report gives bootstrap intervals for the other two
            results[f"interaction_{name}_ci_low"], results[f"interaction_{name}_ci_high"] = row["ci_low"], row["ci_high"]
    results["interaction_no_usage"] = interaction["no_usage_interaction"]
    return results


def step4_cases(xbh) -> dict:
    ranked = cases.pitcher_cases(xbh)
    save(ranked, "p11_pitcher_cases.csv")

    def listing(rows: pd.DataFrame) -> str:
        return " · ".join(f"{name.split(',')[0]} {rate * 100:.0f}%" for name, rate in zip(rows["pitcher_name"], rows["avoidance_rate"]))

    stability = cases.season_stability(xbh)
    return {"cases_top": listing(ranked.head(cases.N_CASES)), "cases_bottom": listing(ranked.tail(cases.N_CASES).iloc[::-1]),
            "omega_squared": stability["omega_squared"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="최종 정리 보고서 수치 재현 파이프라인")
    parser.add_argument("--quick", action="store_true", help="부트스트랩 반복 수를 줄이고 p.10 전체 과정 부트스트랩을 생략")
    parser.add_argument("--steps", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4], help="실행할 단계")
    parser.add_argument("--boot", type=int, default=None, help="p.10 전체 과정 부트스트랩 반복 수 (기본: 전체 200, --quick 0)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("src", "matplotlib", "rpy2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    plots.use_korean_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    budget = dict(QUICK if args.quick else FULL)
    if args.boot is not None:
        budget["interaction_boot"] = args.boot
    full_run = budget == FULL

    started = time.time()
    pitches = load_pitches()
    season_usage = compute_season_usage_rate(pitches)
    xbh = build_xbh_rematch(pitches, season_usage)
    xbh = xbh[xbh["has_next_ab"]].copy()
    logger.info("투구 %s개, 투수 %s명, 재대결 %s건", f"{len(pitches):,}", f"{pitches['pitcher'].nunique():,}", f"{len(xbh):,}")
    use_r = r_available() if {1, 3} & set(args.steps) else False

    results: dict = {}
    if 1 in args.steps:
        logger.info("① 회피 확인 (p.3-5)")
        results.update(step1_avoidance(pitches, season_usage, xbh, budget, use_r))
    if 2 in args.steps:
        logger.info("② 회피 요인 (p.6-7)")
        results.update(step2_factors(pitches, season_usage))
    if 3 in args.steps and use_r:
        logger.info("③ 회피 효과 평가 (p.8-10)")
        results.update(step3_effect(pitches, season_usage, xbh, budget))
    if 4 in args.steps:
        logger.info("④ 사례 (p.11)")
        results.update(step4_cases(xbh))

    check = compare(results, full_run)
    save(check, "report_check.csv")
    (OUT_DIR / "report_check.md").write_text(to_markdown(check) + "\n", encoding="utf-8")
    with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_colwidth", 90):
        print(check.to_string(index=False))
    counts = check["결과"].value_counts()
    print(f"\n보고서 수치 {len(check)}개: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f"  (소요 {(time.time() - started) / 60:.1f}분, 결과: {OUT_DIR})")


if __name__ == "__main__":
    main()
