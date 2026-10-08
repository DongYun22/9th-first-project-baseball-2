"""Situation-matched control ladder (M0 -> M3) for the pitch-avoidance analysis.

Moved out of `notebooks/01_DY_situation_matched_control.ipynb` (same definitions) so the
stage-3 notebooks can reuse it, and generalized to both comparison modes:
- "same_batter": the hit batter's next PA in the same game (main analysis)
- "next_batter": the very next PA (at_bat_number + 1) by the same pitcher (secondary)

Control = `field_out` events drawn per stratum to match the XBH counts, from candidates
that already have a comparison PA in the chosen mode (so neither group loses rows to
eligibility afterwards).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.avoidance_stats import (
    build_stratified_placebo_candidates,
    match_treatment_to_control,
    summarize_diff_in_diff,
)
from src.preprocessing.build_next_ab_dataset import (
    build_event_dataset_for_events,
    filter_to_same_batter_rematch,
    identify_extra_base_hit_events,
)

PLACEBO_OUTCOME_EVENTS = {"field_out"}
PLACEBO_SEED = 20260923
MODES = ("same_batter", "next_batter")

SITUATION_COLUMNS = [
    "runners_n", "risp", "bases", "base_out_state", "inning_bucket", "count_bucket",
    "score_diff", "score_diff_bucket",
]
NEXT_COLUMNS = ["runners_n_next", "risp_next", "outs_next", "score_diff_next"]
PITCH_KEYS = ["game_pk", "at_bat_number", "pitch_number", "pitcher"]

LADDER_STEPS = {
    "M0": ("season", "pitch_family"),
    "M1": ("season", "pitch_family", "base_out_state"),
    "M2": ("season", "pitch_family", "base_out_state", "inning_bucket", "count_bucket"),
    "M3": ("season", "pitch_family", "base_out_state", "inning_bucket", "count_bucket", "score_diff_bucket"),
}
LADDER_LABELS = {
    "M0": "시즌×구종계열",
    "M1": "+주자·아웃",
    "M2": "+이닝·카운트",
    "M3": "+점수차",
}

RUNNER_LEVELS = ["주자 없음", "1루만", "득점권"]
SCORE_LEVELS = ["뒤짐", "동점", "앞섬"]


def add_situation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pitch-level matching variables, as of just before the pitch."""
    out = df.copy()
    on1, on2, on3 = out["on_1b"].notna(), out["on_2b"].notna(), out["on_3b"].notna()
    out["runners_n"] = on1.astype(int) + on2.astype(int) + on3.astype(int)
    out["risp"] = (on2 | on3).astype(int)
    out["bases"] = on1.astype(int) + 2 * on2.astype(int) + 4 * on3.astype(int)
    out["base_out_state"] = out["bases"] * 3 + out["outs_when_up"].astype(int)
    out["inning_bucket"] = np.select([out["inning"] <= 3, out["inning"] <= 6], ["1-3", "4-6"], default="7+")
    out["count_bucket"] = np.select(
        [out["strikes"] > out["balls"], out["strikes"] < out["balls"]], ["pitcher", "batter"], default="even"
    )
    # pitching team's perspective, same as build_next_ab_dataset.compute_score_diff
    out["score_diff"] = np.where(
        out["inning_topbot"] == "Top", out["home_score"] - out["away_score"], out["away_score"] - out["home_score"]
    )
    out["score_diff_bucket"] = np.select(
        [out["score_diff"] < 0, out["score_diff"] > 0], ["뒤짐", "앞섬"], default="동점"
    )
    return out


def filter_to_next_batter(pitches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Rows of `events` whose next PA (at_bat_number + 1) in the same game was thrown by the
    same pitcher -- the eligibility rule of the "next_batter" mode.
    """
    keys = set(
        zip(pitches["game_pk"].astype("int64"), pitches["at_bat_number"].astype("int64"),
            pitches["pitcher"].astype("int64"))
    )
    keep = [
        (int(g), int(ab) + 1, int(p)) in keys
        for g, ab, p in zip(events["game_pk"], events["at_bat_number"], events["pitcher"])
    ]
    return events[pd.Series(keep, index=events.index)]


ELIGIBILITY = {"same_batter": filter_to_same_batter_rematch, "next_batter": filter_to_next_batter}


def attach_season_usage(events: pd.DataFrame, season_usage: pd.DataFrame) -> pd.DataFrame:
    return events.merge(
        season_usage[["pitcher", "season", "pitch_type", "season_usage_rate"]],
        left_on=["pitcher", "season", "hit_pitch_type"],
        right_on=["pitcher", "season", "pitch_type"],
        how="left",
    ).drop(columns=["pitch_type"])


def attach_event_situation(events: pd.DataFrame, pitches_sit: pd.DataFrame) -> pd.DataFrame:
    """Situation of the pitch that produced the event (XBH or control alike)."""
    events = events.drop(columns=[c for c in SITUATION_COLUMNS if c in events.columns])
    situation = pitches_sit[PITCH_KEYS + SITUATION_COLUMNS].rename(columns={"pitch_number": "event_pitch_number"})
    out = events.merge(
        situation, on=["game_pk", "at_bat_number", "event_pitch_number", "pitcher"], how="left", validate="many_to_one"
    )
    assert out[SITUATION_COLUMNS].notna().all().all(), "이벤트 투구를 CSV에서 찾지 못함"
    return out


def attach_next_situation(events: pd.DataFrame, pitches_sit: pd.DataFrame) -> pd.DataFrame:
    """Runners/outs/score at the first pitch of the comparison PA (balance check only)."""
    events = events.drop(columns=[c for c in NEXT_COLUMNS if c in events.columns])
    first = (
        pitches_sit.sort_values(["game_pk", "at_bat_number", "pitcher", "pitch_number"])
        .groupby(["game_pk", "at_bat_number", "pitcher"], as_index=False)
        .first()[["game_pk", "at_bat_number", "pitcher", "runners_n", "risp", "outs_when_up", "score_diff"]]
        .rename(columns={
            "at_bat_number": "next_at_bat_number", "runners_n": "runners_n_next",
            "risp": "risp_next", "outs_when_up": "outs_next", "score_diff": "score_diff_next",
        })
    )
    first["next_at_bat_number"] = first["next_at_bat_number"].astype(float)
    return events.merge(first, on=["game_pk", "next_at_bat_number", "pitcher"], how="left", validate="many_to_one")


def _finish(events: pd.DataFrame, pitches_sit: pd.DataFrame, season_usage: pd.DataFrame) -> pd.DataFrame:
    events = attach_season_usage(events, season_usage)
    return attach_next_situation(attach_event_situation(events, pitches_sit), pitches_sit)


def build_xbh_events(pitches_sit: pd.DataFrame, season_usage: pd.DataFrame, mode: str) -> pd.DataFrame:
    """XBH events that have a comparison PA in `mode`, with all reuse and situation columns."""
    candidates = ELIGIBILITY[mode](pitches_sit, identify_extra_base_hit_events(pitches_sit))
    events = build_event_dataset_for_events(pitches_sit, candidates, next_pa_mode=mode)
    return _finish(events, pitches_sit, season_usage)


def build_step(
    step: str, pitches_sit: pd.DataFrame, xbh: pd.DataFrame, season_usage: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the step's control group and trim XBH to equal per-stratum counts."""
    strata = LADDER_STEPS[step]
    candidates = build_stratified_placebo_candidates(
        pitches_sit, xbh, PLACEBO_OUTCOME_EVENTS, seed=PLACEBO_SEED,
        candidate_filter=ELIGIBILITY[mode], strata_cols=strata,
    )
    placebo = _finish(
        build_event_dataset_for_events(pitches_sit, candidates, next_pa_mode=mode), pitches_sit, season_usage
    )
    xbh_m = match_treatment_to_control(xbh, placebo, strata, seed=PLACEBO_SEED)
    same = xbh_m.groupby(list(strata)).size().sort_index().equals(placebo.groupby(list(strata)).size().sort_index())
    assert same, f"{step}: 칸별 표본 크기가 양쪽에서 다름"
    xbh_m["runner_group"], placebo["runner_group"] = runner_group(xbh_m), runner_group(placebo)
    return xbh_m, placebo


def smd(x: pd.Series, y: pd.Series) -> float:
    """Standardized mean difference (pooled SD). |SMD| < 0.1 reads as balanced."""
    x, y = x.dropna(), y.dropna()
    pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    diff = x.mean() - y.mean()
    if pooled == 0:
        return 0.0 if diff == 0 else float("inf")
    return diff / pooled


def balance_table(xbh: pd.DataFrame, placebo: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = {
        c: {"xbh_mean": xbh[c].mean(), "placebo_mean": placebo[c].mean(), "smd": smd(xbh[c], placebo[c])}
        for c in columns
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def usable(events: pd.DataFrame, baseline_col: str) -> pd.DataFrame:
    d = events[events["has_next_ab"] & events[baseline_col].notna()].copy()
    d["diff"] = d[baseline_col] - d["same_type_share"]
    return d


def pure_reduction(xbh: pd.DataFrame, placebo: pd.DataFrame, baseline_col: str) -> dict:
    """XBH mean of (baseline - comparison-PA share) minus the control's mean, with 95% CI."""
    x, p = usable(xbh, baseline_col), usable(placebo, baseline_col)
    r = summarize_diff_in_diff(x["diff"], p["diff"])
    return {
        "n_xbh": r["n_treatment"], "n_placebo": r["n_control"],
        "xbh_diff": r["treatment_mean"], "placebo_diff": r["control_mean"],
        "net": r["net_effect"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
    }


def runner_group(events: pd.DataFrame) -> np.ndarray:
    return np.select([events["risp"] == 1, events["runners_n"] >= 1], ["득점권", "1루만"], default="주자 없음")


def stratified_reduction(
    xbh: pd.DataFrame, placebo: pd.DataFrame, baseline_col: str, group_col: str, levels: list[str], min_n: int = 30
) -> pd.DataFrame:
    x, p = usable(xbh, baseline_col), usable(placebo, baseline_col)
    rows = []
    for level in levels:
        gx, gp = x[x[group_col] == level], p[p[group_col] == level]
        if len(gx) < min_n or len(gp) < min_n:
            continue
        r = summarize_diff_in_diff(gx["diff"], gp["diff"])
        rows.append({
            group_col: level, "n_xbh": len(gx), "n_placebo": len(gp),
            "net": r["net_effect"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
        })
    return pd.DataFrame(rows)
