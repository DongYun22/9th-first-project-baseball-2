"""Step 1 of the final report (p.3-5): is the hit pitch type avoided in the
rematch, does the game situation explain it, whom is it aimed at, and how
is the type thrown when it is used again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.analysis.batter_specific import (
    event_summary,
    hand_season_usage,
    pa_avoidance,
    pa_table,
    prior_same_batter_avoidance,
)
from src.analysis.hit_quality import build_batted_ball_events
from src.analysis.mixed_effects_model import build_combined_model_dataset
from src.analysis.rematch_execution import deviation_from_typical, group_difference_table, typical_tables
from src.analysis.robust_inference import bootstrap_ci, cluster_bootstrap_did, weighted_did
from src.analysis.situation_matching import LADDER_STEPS, MODES, balance_table, build_step, build_xbh_events

BOOT_SEED = 20261005
XBH_OUTCOMES = ["2·3루타", "홈런"]
ODDS_FORMULA = (
    "reused_same_type ~ group * baseline_usage + balls + strikes + outs_when_up + score_diff "
    "+ stand + pitch_family + factor(season) + (0 + group | pitcher)"
)
ODDS_REQUIRED = ["reused_same_type", "baseline_usage", "balls", "strikes", "outs_when_up", "score_diff",
                 "stand", "pitch_family", "season", "pitcher"]
SLOT_ORDER = [str(k) for k in range(1, 10)] + ["재대결"]


def weighted_ladder(pitches: pd.DataFrame, season_usage: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Pure reduction per mode and ladder step: every eligible field out is
    used as control, weighted to the XBH distribution over the step's strata,
    with a pitcher-cluster bootstrap CI. diff = season usage - comparison-PA share.
    """
    rows = []
    for mode in MODES:
        events = build_batted_ball_events(pitches, season_usage, mode)
        events = events[(events["next_ab_pitch_count"] > 0) & events["season_usage_rate"].notna()]
        events = events.assign(diff=events["season_usage_rate"] - events["same_type_share"])
        xbh, ctrl = events[events["outcome"].isin(XBH_OUTCOMES)], events[events["outcome"] == "아웃"]
        for step, strata in LADDER_STEPS.items():
            r = weighted_did(xbh, ctrl, list(strata))
            boot = cluster_bootstrap_did(xbh, ctrl, list(strata), n_boot=n_boot, seed=BOOT_SEED)
            low, high = bootstrap_ci(r["net"], boot)
            rows.append({"mode": mode, "step": step, "n_xbh": r["n_xbh"], "n_xbh_all": r["n_xbh_all"],
                         "n_ctrl": r["n_ctrl"], "xbh_mean": r["xbh_mean"], "ctrl_mean": r["ctrl_mean"],
                         "net": r["net"], "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def matched_ladder(pitches: pd.DataFrame, season_usage: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The 1:1 matched ladder for the rematch: per step, the runners SMD of
    the matched pair, and the pairs themselves (step -> (xbh, control)).
    """
    xbh = build_xbh_events(pitches, season_usage, "same_batter")
    pairs, rows = {}, []
    for step in LADDER_STEPS:
        xbh_m, control = build_step(step, pitches, xbh, season_usage, "same_batter")
        pairs[step] = (xbh_m, control)
        rows.append({"step": step, "n_xbh": len(xbh_m), "n_control": len(control),
                     "runners_smd": balance_table(xbh_m, control, ["runners_n"]).loc["runners_n", "smd"]})
    return pd.DataFrame(rows), pairs


def ladder_odds_ratios(pairs: dict) -> pd.DataFrame:
    """Odds ratio of throwing the hit type again (XBH vs control) from the
    mixed-effects logistic model, per ladder step. Needs R with lme4.
    """
    from src.analysis.glmer_runner import check_convergence, extract_fixed_effects_table, fit_glmer

    rows = []
    for step, (xbh_m, control) in pairs.items():
        combined = build_combined_model_dataset(
            xbh_m[xbh_m["has_next_ab"] & xbh_m["baseline_usage"].notna()],
            control[control["has_next_ab"] & control["baseline_usage"].notna()],
            required_columns=ODDS_REQUIRED,
        )
        fit_glmer(combined[[*ODDS_REQUIRED, "group"]], ODDS_FORMULA)
        fe = extract_fixed_effects_table().set_index("term")
        rows.append({"step": step, "n": len(combined), "odds_ratio": fe.loc["group", "odds_ratio"],
                     "or_ci_low": fe.loc["group", "or_ci_low"], "or_ci_high": fe.loc["group", "or_ci_high"],
                     "usage_interaction_or": fe.loc["group:baseline_usage", "odds_ratio"],
                     "convergence": check_convergence()})
    return pd.DataFrame(rows)


def _group_diff(df: pd.DataFrame, col: str) -> dict:
    """XBH - control difference of `col` with a pitcher-clustered 95% CI."""
    fit = smf.ols(f"{col} ~ group", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["pitcher"]})
    low, high = fit.conf_int().loc["group"]
    return {"measure": col, "xbh": df.loc[df["group"] == 1, col].mean(), "control": df.loc[df["group"] == 0, col].mean(),
            "diff": fit.params["group"], "ci_low": low, "ci_high": high, "p": fit.pvalues["group"], "n": len(df)}


def batter_specific_tables(pitches: pd.DataFrame, xbh_m3: pd.DataFrame, control_m3: pd.DataFrame):
    """Within the same event: avoidance toward the batters faced before the
    rematch vs toward the rematch batter, each as XBH - control.
    Returns (summary, by_slot, prior) tables.
    """
    events = pd.concat([xbh_m3.assign(group=1), control_m3.assign(group=0)], ignore_index=True)
    events = events[events["has_next_ab"] & events["hit_pitch_type"].notna()].reset_index(drop=True)
    events["event_id"] = np.arange(len(events))
    pas, by_type = pa_table(pitches)
    hand = hand_season_usage(pitches)
    pa = pa_avoidance(events, pas, by_type, hand).merge(events[["event_id", "group"]], on="event_id")
    per_event = event_summary(pa).merge(events[["event_id", "group", "pitcher"]], on="event_id")
    summary = pd.DataFrame([_group_diff(per_event, c) for c in ("avoid_inter", "avoid_rematch", "batter_specific")])

    pa["slot"] = np.where(pa["role"] == "rematch", "재대결", pa["k"].clip(upper=9).astype(str))
    stats = pa.groupby(["slot", "group"])["avoid"].agg(["mean", "std", "size"]).unstack("group")
    rows = []
    for slot in SLOT_ORDER:
        if slot not in stats.index:
            continue
        m1, m0 = stats.loc[slot, ("mean", 1)], stats.loc[slot, ("mean", 0)]
        se = np.sqrt(stats.loc[slot, ("std", 1)] ** 2 / stats.loc[slot, ("size", 1)]
                     + stats.loc[slot, ("std", 0)] ** 2 / stats.loc[slot, ("size", 0)])
        rows.append({"slot": slot, "n_xbh": int(stats.loc[slot, ("size", 1)]), "net": m1 - m0,
                     "ci_low": m1 - m0 - 1.96 * se, "ci_high": m1 - m0 + 1.96 * se})
    by_slot = pd.DataFrame(rows)

    before = per_event.merge(prior_same_batter_avoidance(events, pas, by_type, hand), on="event_id")
    prior = pd.DataFrame([_group_diff(before, c) for c in ("avoid_prior", "avoid_inter", "avoid_rematch")])
    return summary, by_slot, prior


def rematch_execution_tables(pitches: pd.DataFrame, xbh_m3: pd.DataFrame, control_m3: pd.DataFrame):
    """Among M3-matched events where the hit type was thrown again: reuse
    shares per group, and the rematch-PA deviation from the pitcher's norm
    compared between groups. Returns (reuse, differences) tables.
    """
    groups = {"xbh": xbh_m3, "control": control_m3}
    reused = {name: ev[ev["has_next_ab"] & (ev["reused_same_type"] == 1)].reset_index(drop=True)
              for name, ev in groups.items()}
    reuse = pd.DataFrame([{"group": name, "n_matched": len(groups[name]), "n_reused": len(reused[name]),
                           "reuse_share": len(reused[name]) / len(groups[name])} for name in groups])
    base_sum, base_cnt = typical_tables(pitches)
    dev = {name: deviation_from_typical(ev, pitches, base_sum, base_cnt, "rematch") for name, ev in reused.items()}
    return reuse, group_difference_table(dev["xbh"], dev["control"])
