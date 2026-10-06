"""Shared rpy2/lme4::glmer plumbing, factored out so both a single model
run and a multi-model comparison can reuse the exact same fit/extract
logic without duplication.

Not unit tested (R/lme4/rpy2 integration-level, not every contributor has
R installed) -- exercised by actually running the `run_*.py` scripts in
this package against real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

R_CONVERTER = ro.default_converter + numpy2ri.converter + pandas2ri.converter


def fit_glmer(combined: pd.DataFrame, formula: str):
    """Runs the fit as a plain R code string (rather than calling glmer()
    with Python kwargs) because passing an R glmerControl() object through
    rpy2's argument conversion strips its S3 class and breaks glmer's
    internal dispatch -- building the call in R avoids that entirely.
    """
    ro.r("library(lme4)")
    with R_CONVERTER.context():
        r_df = ro.conversion.get_conversion().py2rpy(combined)
        ro.globalenv["model_data"] = r_df
    ro.r(f"""
    model <- glmer(
        {formula},
        data = model_data,
        family = binomial,
        nAGQ = 0,
        control = glmerControl(optimizer = "bobyqa")
    )
    """)
    return ro.globalenv["model"]


def extract_fixed_effects_table() -> pd.DataFrame:
    base = importr("base")
    stats_r = importr("stats")
    coefs = stats_r.coef(base.summary(ro.globalenv["model"]))
    with R_CONVERTER.context():
        df = ro.conversion.get_conversion().rpy2py(base.as_data_frame(coefs))
    df = df.reset_index().rename(
        columns={
            "index": "term",
            "Estimate": "estimate",
            "Std. Error": "std_error",
            "z value": "z_value",
            "Pr(>|z|)": "p_value",
        }
    )
    df["odds_ratio"] = np.exp(df["estimate"])
    df["or_ci_low"] = np.exp(df["estimate"] - 1.96 * df["std_error"])  # Wald 95% CI
    df["or_ci_high"] = np.exp(df["estimate"] + 1.96 * df["std_error"])
    return df


def check_convergence() -> str:
    msg = ro.r("if (is.null(model@optinfo$conv$lme4$messages)) 'OK: no convergence warnings' "
               "else paste(model@optinfo$conv$lme4$messages, collapse='; ')")
    singular = bool(ro.r("isSingular(model)")[0])
    return f"{msg[0]}; isSingular={singular}"


def extract_pitcher_variance_components() -> tuple[float | None, float, float]:
    """Returns (intercept_variance, group_slope_variance, correlation).
    intercept_variance is None when the model has no random intercept term
    (e.g. a `(0 + group | pitcher)` random-effects structure).
    """
    pitcher_vc = ro.r("as.data.frame(VarCorr(model))")
    with R_CONVERTER.context():
        vc_df = ro.conversion.get_conversion().rpy2py(pitcher_vc)
    intercept_rows = vc_df.loc[
        (vc_df["grp"] == "pitcher") & (vc_df["var1"] == "(Intercept)") & vc_df["var2"].isna(), "vcov"
    ]
    intercept_var = float(intercept_rows.iloc[0]) if len(intercept_rows) else None
    group_var = float(
        vc_df.loc[(vc_df["grp"] == "pitcher") & (vc_df["var1"] == "group") & vc_df["var2"].isna(), "vcov"].iloc[0]
    )
    corr_row = vc_df.loc[(vc_df["grp"] == "pitcher") & (vc_df["var2"] == "group")]
    corr = float(corr_row["sdcor"].iloc[0]) if len(corr_row) else float("nan")
    return intercept_var, group_var, corr


def extract_variance_components(grouping: str = "pitcher") -> dict[str, float | None]:
    """Generic (1 | grouping) random-intercept variance plus residual
    variance, for a model fit by either fit_glmer or fit_lmer. Returns
    {f"{grouping}_intercept": ..., "residual": ...}; "residual" is None for
    a binomial glmer fit (VarCorr has no Residual row for those), and
    present for a gaussian lmer fit.
    """
    vc = ro.r("as.data.frame(VarCorr(model))")
    with R_CONVERTER.context():
        vc_df = ro.conversion.get_conversion().rpy2py(vc)
    intercept_rows = vc_df.loc[
        (vc_df["grp"] == grouping) & (vc_df["var1"] == "(Intercept)") & vc_df["var2"].isna(), "vcov"
    ]
    residual_rows = vc_df.loc[vc_df["grp"] == "Residual", "vcov"]
    return {
        f"{grouping}_intercept": float(intercept_rows.iloc[0]) if len(intercept_rows) else None,
        "residual": float(residual_rows.iloc[0]) if len(residual_rows) else None,
    }


def extract_random_effects_table() -> pd.DataFrame:
    pitcher_ranef = ro.r("as.data.frame(ranef(model)$pitcher)")
    with R_CONVERTER.context():
        df = ro.conversion.get_conversion().rpy2py(pitcher_ranef)
    df = df.reset_index().rename(columns={"index": "pitcher", "(Intercept)": "re_intercept", "group": "re_group"})
    return df


def fit_lmer(data: pd.DataFrame, formula: str):
    """Linear mixed model (REML) via lme4::lmer, run as an R code string for
    the same reason as fit_glmer.
    """
    ro.r("library(lme4)")
    with R_CONVERTER.context():
        r_df = ro.conversion.get_conversion().py2rpy(data)
        ro.globalenv["model_data"] = r_df
    ro.r(f"model <- lmer({formula}, data = model_data, REML = TRUE)")
    return ro.globalenv["model"]


def extract_lmer_fixed_effects() -> pd.DataFrame:
    """Fixed effects of the current lmer model with Wald (normal) p-values and
    95% CIs -- lmer reports no p-values, and with tens of thousands of rows
    the t distribution is indistinguishable from the normal.
    """
    from scipy import stats

    coefs = ro.r("as.data.frame(summary(model)$coefficients)")
    with R_CONVERTER.context():
        df = ro.conversion.get_conversion().rpy2py(coefs)
    df = df.reset_index().rename(
        columns={"index": "term", "Estimate": "estimate", "Std. Error": "std_error", "t value": "t_value"}
    )
    df["ci_low"] = df["estimate"] - 1.96 * df["std_error"]
    df["ci_high"] = df["estimate"] + 1.96 * df["std_error"]
    df["p_value"] = 2 * stats.norm.sf(df["t_value"].abs())
    return df


def extract_fixed_effects_vcov() -> pd.DataFrame:
    """Variance-covariance matrix of the current model's fixed effects,
    labelled by term on both axes.
    """
    terms = list(ro.r("names(fixef(model))"))
    with R_CONVERTER.context():
        matrix = np.asarray(ro.conversion.get_conversion().rpy2py(ro.r("as.matrix(vcov(model))")))
    return pd.DataFrame(matrix, index=terms, columns=terms)


def extract_design_means() -> pd.Series:
    """Column means of the current model's fixed-effects design matrix."""
    terms = list(ro.r("names(fixef(model))"))
    with R_CONVERTER.context():
        means = np.asarray(ro.conversion.get_conversion().rpy2py(ro.r("colMeans(model.matrix(model))")))
    return pd.Series(means, index=terms)


def extract_ranef_table(grouping: str = "pitcher") -> pd.DataFrame:
    """Per-group random effects of the current model; columns are named
    `re_<term>` (intercept -> re_intercept).
    """
    table = ro.r(f"as.data.frame(ranef(model)${grouping})")
    with R_CONVERTER.context():
        df = ro.conversion.get_conversion().rpy2py(table)
    df = df.reset_index().rename(columns={"index": grouping, "(Intercept)": "re_intercept"})
    return df.rename(columns={c: f"re_{c}" for c in df.columns if c not in (grouping, "re_intercept")})


def extract_predictions() -> np.ndarray:
    preds = ro.r('predict(model, type="response")')
    with R_CONVERTER.context():
        arr = ro.conversion.get_conversion().rpy2py(preds)
    return np.asarray(arr)
