"""Mixed-effects logistic regression for the pitch-avoidance question,
fit with R's lme4::glmer via rpy2 (statsmodels' MixedLM does not support a
binomial/logistic response with random effects).

Pure data-prep / post-processing helpers are ordinary, unit-tested Python
functions. Anything that touches R (fitting, extracting model objects) is
integration-level and is exercised by actually running
`run_mixed_effects_model.py` against real data, not by unit tests -- not
every contributor on this project has R installed.
"""

from __future__ import annotations

import math

import pandas as pd

REQUIRED_MODEL_COLUMNS = [
    "reused_same_type",
    "baseline_usage",
    "balls",
    "strikes",
    "outs_when_up",
    "score_diff",
    "stand",
    "pitch_family",
    "season",
    "pitcher",
]

# The first model fit: within has_next_ab=True (same pitcher, very next
# batter, half-inning never ended), the catcher is essentially never
# substituted -- only 5 of 123,044 rows had catcher_changed=1, so its
# coefficient came out practically unidentifiable (estimate -0.43, SE 0.98).
# Also came out isSingular=TRUE: the random-intercept variance for pitcher
# was estimated at the zero boundary.
MODEL_FORMULA_ORIGINAL = (
    "reused_same_type ~ group * baseline_usage + balls + strikes + outs_when_up "
    "+ score_diff + stand + pitch_family + factor(season) + catcher_changed "
    "+ (1 + group | pitcher)"
)

# Robustness check: drop catcher_changed (unidentifiable), and simplify the
# random-effects structure to a random slope only -- no random intercept --
# since the original model's intercept variance was at the zero boundary.
MODEL_FORMULA_ROBUSTNESS = (
    "reused_same_type ~ group * baseline_usage + balls + strikes + outs_when_up "
    "+ score_diff + stand + pitch_family + factor(season) "
    "+ (0 + group | pitcher)"
)

MODEL_FORMULA = MODEL_FORMULA_ROBUSTNESS  # current recommended default

# The robustness spec plus platoon (same-handed batter/pitcher = 1): the main
# effect and its interaction with group test whether the avoidance effect
# differs for same- vs opposite-handed matchups. Needs `platoon_match` in the
# data (pass required_columns=REQUIRED_MODEL_COLUMNS + ["platoon_match"]).
MODEL_FORMULA_PLATOON = (
    "reused_same_type ~ group * baseline_usage + platoon_match + group:platoon_match "
    "+ balls + strikes + outs_when_up + score_diff + stand + pitch_family + factor(season) "
    "+ (0 + group | pitcher)"
)


def build_combined_model_dataset(
    xbh_events: pd.DataFrame,
    placebo_events: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Stack the XBH (group=1) and placebo (group=0) event datasets into one
    modeling-ready frame, dropping any row missing a required covariate
    (glmer's default na.omit would do this silently; we do it explicitly so
    the row count is reportable) and coercing `pitcher` to a string grouping
    factor. `required_columns` defaults to REQUIRED_MODEL_COLUMNS.
    """
    xbh = xbh_events.assign(group=1)
    placebo = placebo_events.assign(group=0)
    combined = pd.concat([xbh, placebo], ignore_index=True)
    combined = combined.dropna(subset=required_columns or REQUIRED_MODEL_COLUMNS).copy()
    combined["pitcher"] = combined["pitcher"].astype(str)
    return combined


def compare_random_effects(a: pd.DataFrame, b: pd.DataFrame, column: str = "re_group") -> dict:
    """Pearson/Spearman correlation of a per-pitcher random-effect column
    between two model fits, over the pitchers present in both.
    """
    merged = a[["pitcher", column]].merge(b[["pitcher", column]], on="pitcher", suffixes=("_a", "_b"))
    return {
        "n_common": len(merged),
        "pearson": merged[f"{column}_a"].corr(merged[f"{column}_b"], method="pearson"),
        "spearman": merged[f"{column}_a"].corr(merged[f"{column}_b"], method="spearman"),
    }


def compute_icc(pitcher_intercept_variance: float) -> float:
    """Intraclass correlation for the random-intercept variance component of
    a logistic mixed model, using the standard latent-variable residual
    variance pi^2/3 for the binomial/logit link.
    """
    return pitcher_intercept_variance / (pitcher_intercept_variance + (math.pi**2) / 3)


def compute_icc_gaussian(intercept_var: float, residual_var: float) -> float:
    """Intraclass correlation for a gaussian (lmer) random-intercept model:
    intercept variance over intercept + residual variance, vs. compute_icc's
    logistic-link pi^2/3 formula.
    """
    return intercept_var / (intercept_var + residual_var)


def rank_pitchers_by_group_slope(
    random_effects: pd.DataFrame, pitcher_meta: pd.DataFrame, n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`random_effects`: columns [pitcher, re_intercept, re_group] (one row
    per pitcher's estimated random effects). `pitcher_meta`: columns
    [pitcher, pitcher_name, n_xbh_events]. Returns (top_n, bottom_n) by
    re_group, each with name and sample size attached.
    """
    merged = random_effects.merge(pitcher_meta, on="pitcher", how="left").sort_values("re_group")
    bottom = merged.head(n).reset_index(drop=True)
    top = merged.tail(n).sort_values("re_group", ascending=False).reset_index(drop=True)
    return top, bottom


def compute_calibration_table(observed: pd.Series, predicted: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Bin events into `n_bins` deciles of predicted probability and compare
    each bin's mean predicted probability to its mean observed outcome.
    """
    bins = pd.qcut(predicted, n_bins, duplicates="drop")
    table = pd.DataFrame({"observed": observed.to_numpy(), "predicted": predicted.to_numpy(), "bin": bins})
    return (
        table.groupby("bin", observed=True)
        .agg(n=("observed", "size"), mean_predicted=("predicted", "mean"), mean_observed=("observed", "mean"))
        .reset_index()
    )
