"""Pre-rematch moderators for "when does avoiding the hit pitch pay off?".

Both moderators are built only from pitches thrown on dates strictly before
the rematch's game, so the avoidance decision cannot leak into them:
- hit pitch reliance: the pitcher's prior usage share of the pitch type that
  was hit (and whether it was his most-used pitch);
- alternative quality: prior run value above expected (RVAE, lower = better
  for the pitcher) of his other pitch types, shrunk toward 0 with an
  empirical-Bayes weight n / (n + k) so a few good PA-ending pitches of a
  rarely used pitch type are not taken at face value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_shrinkage_k(rvae_pitches: pd.DataFrame, min_n: int = 5) -> float:
    """Method-of-moments k = sigma^2 / tau^2 for shrinking a (pitcher,
    pitch_type) mean RVAE toward 0. `rvae_pitches` needs pitcher, pitch_type,
    rvae (PA-ending pitches only). sigma^2 is the pooled within-group
    variance; tau^2 is the between-group variance of true means, i.e.
    Var(group means) - sigma^2 * mean(1/n).
    """
    g = rvae_pitches.groupby(["pitcher", "pitch_type"])["rvae"].agg(["count", "mean", "var"])
    g = g[g["count"] >= min_n]
    sigma2 = float((g["var"] * (g["count"] - 1)).sum() / (g["count"] - 1).sum())
    tau2 = float(np.mean(g["mean"] ** 2) - sigma2 * np.mean(1.0 / g["count"]))
    if tau2 <= 0:
        raise ValueError("between-group variance estimated <= 0; shrinkage k undefined")
    return sigma2 / tau2


def _daily_cumulative(pitches: pd.DataFrame) -> pd.DataFrame:
    p = pitches.dropna(subset=["pitch_type"])
    daily = (
        p.groupby(["pitcher", "pitch_type", "game_date"])
        .agg(n=("pitch_type", "size"), n_dec=("rvae", "count"), rvae_sum=("rvae", "sum"))
        .reset_index()
        .sort_values("game_date")
    )
    for col in ["n", "n_dec", "rvae_sum"]:
        daily[f"cum_{col}"] = daily.groupby(["pitcher", "pitch_type"])[col].cumsum()
    return daily[["pitcher", "pitch_type", "game_date", "cum_n", "cum_n_dec", "cum_rvae_sum"]]


def add_prior_pitch_features(
    events: pd.DataFrame,
    pitches: pd.DataFrame,
    shrinkage_k: float,
    min_alt_usage: float = 0.05,
    min_prior_pitches: int = 200,
) -> pd.DataFrame:
    """Append prior-only moderators to `events` (needs pitcher, game_date,
    hit_pitch_type). `pitches` needs pitcher, game_date, pitch_type and rvae
    (NaN unless the pitch ended the PA). Adds: n_prior_pitches,
    hit_prior_usage, hit_is_primary, n_alternatives, alt_rvae_best (lowest
    shrunk RVAE among alternatives = best), alt_rvae_wmean (usage-weighted).
    Events with fewer than `min_prior_pitches` prior pitches, or no
    alternative above `min_alt_usage`, get NaN moderators.
    """
    ev = events.reset_index(drop=True).copy()
    ev["_eid"] = np.arange(len(ev))
    cum = _daily_cumulative(pitches)

    types = pitches.dropna(subset=["pitch_type"])[["pitcher", "pitch_type"]].drop_duplicates()
    left = ev[["_eid", "pitcher", "game_date", "hit_pitch_type"]].merge(types, on="pitcher")
    left = left.sort_values("game_date")
    prior = pd.merge_asof(
        left, cum, on="game_date", by=["pitcher", "pitch_type"], allow_exact_matches=False
    )
    prior[["cum_n", "cum_n_dec", "cum_rvae_sum"]] = prior[["cum_n", "cum_n_dec", "cum_rvae_sum"]].fillna(0)
    prior = prior[prior["cum_n"] > 0].copy()

    total = prior.groupby("_eid")["cum_n"].transform("sum")
    prior["usage"] = prior["cum_n"] / total
    prior["shrunk"] = prior["cum_rvae_sum"] / (prior["cum_n_dec"] + shrinkage_k)
    prior["total"] = total
    prior["is_hit"] = prior["pitch_type"] == prior["hit_pitch_type"]

    hit = prior[prior["is_hit"]].set_index("_eid")
    max_usage = prior.groupby("_eid")["usage"].max()
    alts = prior[~prior["is_hit"] & (prior["usage"] >= min_alt_usage)]
    alt_agg = alts.groupby("_eid").apply(
        lambda d: pd.Series({
            "n_alternatives": len(d),
            "alt_rvae_best": d["shrunk"].min(),
            "alt_rvae_wmean": float(np.average(d["shrunk"], weights=d["usage"])),
        }),
        include_groups=False,
    )
    feats = pd.DataFrame(index=ev["_eid"])
    feats["n_prior_pitches"] = prior.groupby("_eid")["total"].first()
    feats["hit_prior_usage"] = hit["usage"]
    feats["hit_is_primary"] = (hit["usage"] >= max_usage.reindex(hit.index)).astype(float).reindex(feats.index)
    feats = feats.join(alt_agg)
    feats["n_alternatives"] = feats["n_alternatives"].fillna(0)
    feats["n_prior_pitches"] = feats["n_prior_pitches"].fillna(0)

    too_thin = feats["n_prior_pitches"] < min_prior_pitches
    feats.loc[too_thin, ["hit_prior_usage", "hit_is_primary", "alt_rvae_best", "alt_rvae_wmean"]] = np.nan
    feats.loc[too_thin, "n_alternatives"] = 0
    feats.loc[feats["n_alternatives"] == 0, ["alt_rvae_best", "alt_rvae_wmean"]] = np.nan

    out = pd.concat([ev.drop(columns="_eid"), feats.reset_index(drop=True)], axis=1)
    return out


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in input order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()
