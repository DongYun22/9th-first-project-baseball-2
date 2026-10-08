"""How the hit pitch type is thrown when it is used again in the rematch
(final report p.4-5): each rematch PA's location, speed and movement as a
deviation from that pitcher's own norm, compared between rematches after an
extra-base hit and rematches after a field out (the M3-matched control).

The norm is the pitcher's season mean for the same pitch type against the
same batter hand, leaving the PA itself out, and only when at least
MIN_BASELINE_PITCHES other pitches define it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.avoidance_stats import summarize_diff_in_diff

MEASURES = ["plate_x", "plate_z", "pfx_x", "pfx_z", "release_speed", "release_spin_rate", "release_extension"]
# measure -> (label, multiplier to the display unit; pfx and extension are stored in feet)
DELTA_LABELS = {
    "plate_x": ("코스 x (ft)", 1), "plate_z": ("코스 z (ft)", 1),
    "pfx_x": ("무브먼트 x (in)", 12), "pfx_z": ("무브먼트 z (in)", 12),
    "release_speed": ("구속 (mph)", 1), "release_spin_rate": ("회전수 (rpm)", 1),
    "release_extension": ("익스텐션 (in)", 12),
}
BASE_KEYS = ["pitcher", "season", "pitch_type", "stand"]
MIN_BASELINE_PITCHES = 30
MIN_EVENTS_BOTH = 50


def typical_tables(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum and count of each measure per pitcher x season x pitch type x batter hand."""
    grouped = pitches.dropna(subset=["pitch_type"]).groupby(BASE_KEYS)[MEASURES]
    return grouped.sum(min_count=1), grouped.count()


def deviation_from_typical(
    events: pd.DataFrame, pitches: pd.DataFrame, base_sum: pd.DataFrame, base_cnt: pd.DataFrame, phase: str
) -> pd.DataFrame:
    """Per event, the mean deviation from the pitcher's norm of the hit-type
    pitches in the rematch PA (phase="rematch") or in the event PA with the
    event pitch itself removed (phase="event").
    """
    ev = events[["game_pk", "pitcher", "season", "hit_pitch_type", "p_throws", "at_bat_number",
                 "event_pitch_number", "next_at_bat_number"]].copy()
    ev["event_id"] = np.arange(len(ev))
    ev["target_at_bat"] = ev["next_at_bat_number"].astype(int) if phase == "rematch" else ev["at_bat_number"]
    ev = ev.drop(columns=["at_bat_number", "next_at_bat_number"])
    px = pitches.dropna(subset=["pitch_type"])[
        ["game_pk", "pitcher", "season", "pitch_type", "stand", "at_bat_number", "pitch_number", *MEASURES]
    ]
    joined = ev.merge(
        px, left_on=["game_pk", "pitcher", "season", "hit_pitch_type", "target_at_bat"],
        right_on=["game_pk", "pitcher", "season", "pitch_type", "at_bat_number"],
    )
    if phase == "event":
        joined = joined[joined["pitch_number"] != joined["event_pitch_number"]]
    grouped = joined.groupby(["event_id", *BASE_KEYS, "p_throws"])
    pa_sum, pa_cnt = grouped[MEASURES].sum(min_count=1), grouped[MEASURES].count()
    keys = pd.MultiIndex.from_frame(pa_sum.reset_index()[BASE_KEYS])
    rest_cnt = base_cnt.reindex(keys).to_numpy() - pa_cnt.to_numpy()
    rest_mean = (base_sum.reindex(keys).to_numpy() - pa_sum.fillna(0).to_numpy()) / np.where(
        rest_cnt >= MIN_BASELINE_PITCHES, rest_cnt, np.nan
    )
    pa_mean = pa_sum.to_numpy() / np.where(pa_cnt.to_numpy() > 0, pa_cnt.to_numpy(), np.nan)
    dev = pd.DataFrame(pa_mean - rest_mean, columns=MEASURES, index=pa_sum.index).reset_index()
    return dev.groupby("event_id").agg(
        {**{m: "mean" for m in MEASURES}, "p_throws": "first", "pitch_type": "first"}
    ).rename(columns={"pitch_type": "hit_pitch_type"})


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    """Mean and its normal-approximation 95% CI."""
    s = values.dropna()
    n = len(s)
    m = s.mean()
    half = 1.96 * s.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half


def group_difference_table(
    dev_xbh: pd.DataFrame, dev_control: pd.DataFrame, min_events: int = MIN_EVENTS_BOTH
) -> pd.DataFrame:
    """Long table per pitcher hand x pitch type x measure: each group's mean
    deviation, their difference (XBH - control) and its 95% CI. Cells where
    either group has fewer than `min_events` events are skipped. The CI
    treats events as independent (no pitcher clustering, no multiplicity
    correction).
    """
    rows = []
    for (hand, pitch_type), gx in dev_xbh.groupby(["p_throws", "hit_pitch_type"]):
        gc = dev_control[(dev_control["p_throws"] == hand) & (dev_control["hit_pitch_type"] == pitch_type)]
        if len(gx) < min_events or len(gc) < min_events:
            continue
        for measure, (label, scale) in DELTA_LABELS.items():
            x, c = (gx[measure] * scale).dropna(), (gc[measure] * scale).dropna()
            r = summarize_diff_in_diff(x, c, alternative="two-sided")
            mx, lx, hx = mean_ci(x)
            mc, lc, hc = mean_ci(c)
            rows.append({
                "p_throws": hand, "hit_pitch_type": pitch_type, "measure": measure, "label": label,
                "n_xbh": len(gx), "n_control": len(gc),
                "xbh": mx, "xbh_ci_low": lx, "xbh_ci_high": hx,
                "control": mc, "control_ci_low": lc, "control_ci_high": hc,
                "diff": r["net_effect"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                "distinct": bool(r["ci_low"] > 0 or r["ci_high"] < 0),
            })
    return pd.DataFrame(rows)
