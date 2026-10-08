"""Statistical helpers for testing whether pitchers avoid re-throwing the
pitch type that was just hit for extra bases.

Three tiers of rigor, from weakest to strongest control for noise/regression
to the mean:
1. Paired comparison against the within-game baseline_usage column.
2. Paired comparison against each pitcher's season-level usage rate of the
   pitch type (much less noisy: hundreds/thousands of pitches instead of a
   handful).
3. A placebo/diff-in-diff comparison against a control outcome (e.g.
   field_out) built through the exact same next-at-bat logic as the real
   extra-base-hit events, stratified to match their (season, pitch_family)
   distribution.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.preprocessing.build_next_ab_dataset import identify_events_by_outcome
from src.preprocessing.pitch_family import map_pitch_family


def compute_season_usage_rate(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per (pitcher, season, pitch_type): share of that pitcher's season's
    pitches that were this pitch type -- a low-noise baseline in place of
    the within-game `baseline_usage` column. Pitches with no pitch_type are
    left out of both counts and totals (they belong to no type, and would
    otherwise give a missing-type event a bogus "usage rate" to match).
    """
    known = pitches.dropna(subset=["pitch_type"])
    counts = known.groupby(["pitcher", "season", "pitch_type"]).size().rename("n").reset_index()
    totals = known.groupby(["pitcher", "season"]).size().rename("total").reset_index()
    usage = counts.merge(totals, on=["pitcher", "season"])
    usage["season_usage_rate"] = usage["n"] / usage["total"]
    return usage


def build_stratified_placebo_candidates(
    pitches: pd.DataFrame,
    treatment_events: pd.DataFrame,
    outcome_events: set[str],
    seed: int,
    candidate_filter: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] | None = None,
    strata_cols: tuple[str, ...] = ("season", "pitch_family"),
) -> pd.DataFrame:
    """Sample pitch-level rows whose `events` is in `outcome_events` (e.g.
    {"field_out"}), stratified to match `treatment_events`' distribution over
    `strata_cols` exactly -- a control group that differs from the treatment
    group only in whether the pitch was hit for extra bases, not in the
    matched dimensions. The default (season, pitch_family) is the main
    analysis' matching; adding "pitcher" gives same-pitcher matching.

    `treatment_events` must have every `strata_cols` column, including
    `pitch_family` (rows with a null `pitch_family` are dropped from the
    quota). If a stratum's candidate pool is smaller than its quota, every
    available candidate in that stratum is used instead of raising; strata
    with candidates but no treatment events are never sampled.

    `candidate_filter(pitches, candidates)` optionally narrows the candidate
    pool BEFORE sampling (e.g. to same-batter-rematch-eligible rows), so the
    control group is matched on the treatment group's eligible strata rather
    than losing rows to eligibility afterwards.
    """
    strata = list(strata_cols)
    quota = treatment_events.dropna(subset=["pitch_family"]).groupby(strata).size()

    candidates = identify_events_by_outcome(pitches, outcome_events).copy()
    if candidate_filter is not None:
        candidates = candidate_filter(pitches, candidates).copy()
    candidates["pitch_family"] = candidates["pitch_type"].map(map_pitch_family)
    candidates = candidates.dropna(subset=["pitch_family"])

    pools = {
        (key if isinstance(key, tuple) else (key,)): positions
        for key, positions in candidates.groupby(strata).indices.items()
    }

    rng = np.random.default_rng(seed)
    sampled_parts = []
    for key, n_needed in quota.items():
        positions = pools.get(key if isinstance(key, tuple) else (key,))
        if positions is None:
            continue
        pool = candidates.iloc[positions]
        if len(pool) <= n_needed:
            sampled_parts.append(pool)
        else:
            idx = rng.choice(pool.index.to_numpy(), size=n_needed, replace=False)
            sampled_parts.append(pool.loc[idx])

    return pd.concat(sampled_parts).copy()


def match_treatment_to_control(
    treatment_events: pd.DataFrame,
    control_events: pd.DataFrame,
    strata_cols: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    """Trim `treatment_events` so that, in every stratum, it has exactly as
    many rows as `control_events` (random subsample when it has more; strata
    with no controls are dropped). Used after a same-pitcher control draw
    that could not fill every treatment stratum, so both groups end up with
    identical stratum counts.
    """
    strata = list(strata_cols)
    control_counts = control_events.groupby(strata).size()
    rng = np.random.default_rng(seed)
    kept = []
    for key, group in treatment_events.groupby(strata):
        n_control = control_counts.get(key, 0)
        if n_control == 0:
            continue
        if len(group) <= n_control:
            kept.append(group)
        else:
            kept.append(group.iloc[rng.choice(len(group), size=n_control, replace=False)])
    return pd.concat(kept).copy()


def compute_expected_reuse_prob(baseline_usage: pd.Series, next_ab_pitch_count: pd.Series) -> pd.Series:
    """P(>=1 reuse) under a null of 'pitcher keeps throwing at baseline_usage
    rate, independently, on each of next_ab_pitch_count pitches'.
    """
    return 1 - (1 - baseline_usage) ** next_ab_pitch_count


def summarize_avoidance_by_level(
    xbh: pd.DataFrame,
    placebo: pd.DataFrame,
    by: str,
    baseline_col: str,
    min_n: int,
) -> pd.DataFrame:
    """One row per level of `by` (e.g. hit_pitch_type) with the placebo-netted
    usage drop and the raw/expected-adjusted reuse comparison. Levels need at
    least `min_n` usable events (has_next_ab and a non-null baseline) in BOTH
    groups. `net_drop` = XBH drop - placebo drop, where drop = baseline -
    same_type_share; `reuse_gap_net` compares (observed - expected reuse under
    the baseline) between the groups, so it is comparable across levels with
    different baseline usage.
    """

    def usable(events: pd.DataFrame) -> pd.DataFrame:
        d = events[events["has_next_ab"] & events[baseline_col].notna()].copy()
        d["drop"] = d[baseline_col] - d["same_type_share"]
        d["reuse_gap"] = d["reused_same_type"] - compute_expected_reuse_prob(d[baseline_col], d["next_ab_pitch_count"])
        return d

    x, p = usable(xbh), usable(placebo)
    rows = []
    for level in sorted(set(x[by].dropna().unique()) & set(p[by].dropna().unique())):
        gx, gp = x[x[by] == level], p[p[by] == level]
        if len(gx) < min_n or len(gp) < min_n:
            continue
        drop = summarize_diff_in_diff(gx["drop"], gp["drop"])
        gap = summarize_diff_in_diff(gx["reuse_gap"], gp["reuse_gap"], alternative="less")
        rows.append({
            by: level,
            "n_xbh": len(gx), "n_placebo": len(gp),
            "xbh_baseline": gx[baseline_col].mean(), "xbh_post": gx["same_type_share"].mean(),
            "placebo_baseline": gp[baseline_col].mean(), "placebo_post": gp["same_type_share"].mean(),
            "xbh_drop": drop["treatment_mean"], "placebo_drop": drop["control_mean"],
            "net_drop": drop["net_effect"], "net_ci_low": drop["ci_low"], "net_ci_high": drop["ci_high"],
            "reuse_xbh": gx["reused_same_type"].mean(), "reuse_placebo": gp["reused_same_type"].mean(),
            "reuse_gap_net": gap["net_effect"],
        })
    return pd.DataFrame(rows)


def relative_reduction(
    xbh_baseline: float, xbh_post: float, placebo_baseline: float, placebo_post: float
) -> float:
    """Share of the expected post-hit usage that is lost. The expectation applies
    the placebo's own change to the XBH group's baseline (what the XBH group
    would have gone to had the hit not mattered): expected = xbh_baseline -
    (placebo_baseline - placebo_post); result = 1 - xbh_post / expected.
    NaN if the expectation is not positive.
    """
    expected_post = xbh_baseline - (placebo_baseline - placebo_post)
    if expected_post <= 0:
        return float("nan")
    return 1 - xbh_post / expected_post


def bootstrap_relative_reduction_ci(
    xbh_baseline: np.ndarray,
    xbh_post: np.ndarray,
    placebo_baseline: np.ndarray,
    placebo_post: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of `relative_reduction`, resampling events
    (baseline and post stay paired within an event) separately per group.
    """
    xb, xp = np.asarray(xbh_baseline, dtype=float), np.asarray(xbh_post, dtype=float)
    pb, pp = np.asarray(placebo_baseline, dtype=float), np.asarray(placebo_post, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, len(xb), len(xb))
        ip = rng.integers(0, len(pb), len(pb))
        estimates[i] = relative_reduction(xb[ix].mean(), xp[ix].mean(), pb[ip].mean(), pp[ip].mean())
    return (
        float(np.nanpercentile(estimates, 100 * alpha / 2)),
        float(np.nanpercentile(estimates, 100 * (1 - alpha / 2))),
    )


def summarize_reuse_rate_reduction(
    xbh: pd.DataFrame,
    placebo: pd.DataFrame,
    by: str,
    baseline_col: str,
    levels: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per level of `by`, in the order given by `levels` (levels missing from
    either group are skipped): how far the re-throw usage rate (same_type_share)
    falls from the baseline usage rate, for the XBH and placebo groups, the
    placebo-netted drop (%p, with CI), the relative reduction (with bootstrap
    CI), and the binary reuse rate (re-thrown at least once) difference.
    """

    def usable(events: pd.DataFrame) -> pd.DataFrame:
        return events[events["has_next_ab"] & events[baseline_col].notna()]

    x, p = usable(xbh), usable(placebo)
    rows = []
    for level in levels:
        gx, gp = x[x[by] == level], p[p[by] == level]
        if gx.empty or gp.empty:
            continue
        xb, xp = gx[baseline_col].to_numpy(), gx["same_type_share"].to_numpy()
        pb, pp = gp[baseline_col].to_numpy(), gp["same_type_share"].to_numpy()
        drop = summarize_diff_in_diff(pd.Series(xb - xp), pd.Series(pb - pp))
        rel = relative_reduction(xb.mean(), xp.mean(), pb.mean(), pp.mean())
        rel_low, rel_high = bootstrap_relative_reduction_ci(xb, xp, pb, pp, n_boot=n_boot, seed=seed)
        r1, r2 = gx["reused_same_type"].mean(), gp["reused_same_type"].mean()
        se = np.sqrt(r1 * (1 - r1) / len(gx) + r2 * (1 - r2) / len(gp))
        rows.append({
            by: level, "n_xbh": len(gx), "n_placebo": len(gp),
            "xbh_baseline": xb.mean(), "xbh_post": xp.mean(), "xbh_drop": drop["treatment_mean"],
            "placebo_baseline": pb.mean(), "placebo_post": pp.mean(), "placebo_drop": drop["control_mean"],
            "net_drop": drop["net_effect"], "net_ci_low": drop["ci_low"], "net_ci_high": drop["ci_high"],
            "relative_reduction": rel, "rel_ci_low": rel_low, "rel_ci_high": rel_high,
            "reuse_xbh": r1, "reuse_placebo": r2, "reuse_diff": r1 - r2,
            "reuse_diff_ci_low": r1 - r2 - 1.96 * se, "reuse_diff_ci_high": r1 - r2 + 1.96 * se,
        })
    return pd.DataFrame(rows)


def summarize_paired_diff(diff: pd.Series, alpha: float = 0.05) -> dict:
    """One-sample summary of a paired difference against 0: mean, normal-
    approximation CI, one-sample t-test, and Wilcoxon signed-rank test.
    """
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = mean - z * se, mean + z * se
    t_stat, t_p = stats.ttest_1samp(diff, 0)
    w_stat, w_p = stats.wilcoxon(diff)
    return {
        "n": n,
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "t_p": t_p,
        "w_stat": w_stat,
        "w_p": w_p,
    }


def summarize_diff_in_diff(
    d_treatment: pd.Series,
    d_control: pd.Series,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> dict:
    """Compare two independent samples of a per-event difference metric
    (e.g. baseline_usage - same_type_share), returning each group's mean,
    the net (treatment - control) effect, its normal-approximation CI, and
    a Welch t-test / Mann-Whitney U test in the given one-sided direction.
    """
    n1, n2 = len(d_treatment), len(d_control)
    m1, m2 = d_treatment.mean(), d_control.mean()
    net = m1 - m2
    se = np.sqrt(d_treatment.var(ddof=1) / n1 + d_control.var(ddof=1) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = net - z * se, net + z * se
    t_stat, t_p = stats.ttest_ind(d_treatment, d_control, equal_var=False, alternative=alternative)
    u_stat, u_p = stats.mannwhitneyu(d_treatment, d_control, alternative=alternative)
    return {
        "n_treatment": n1,
        "n_control": n2,
        "treatment_mean": m1,
        "control_mean": m2,
        "net_effect": net,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "t_p": t_p,
        "u_stat": u_stat,
        "u_p": u_p,
    }
