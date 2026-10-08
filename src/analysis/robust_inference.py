"""Seed-free control estimates and pitcher-cluster bootstrap CIs for the avoidance analysis.

The matched-control ladder (situation_matching) draws ONE 1:1 control sample with one seed and
reports CIs that treat events as independent. Here:
- weighted_did: use EVERY eligible control in a stratum, weighted so each stratum counts as
  often as it does among the XBH events (no sampling, so no seed);
- cluster_bootstrap_did: resample pitchers with replacement, so repeated events of the same
  pitcher/game no longer narrow the CI;
- sample_1to1_net: the original 1:1 draw, vectorized, to show how much the seed alone moves it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _stratum_codes(xbh: pd.DataFrame, ctrl: pd.DataFrame, strata: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    keys = pd.concat([xbh[strata], ctrl[strata]], ignore_index=True)
    codes = keys.groupby(strata, sort=False, dropna=False).ngroup().to_numpy()
    return codes[: len(xbh)], codes[len(xbh):], int(codes.max()) + 1


def _net_from_totals(nx: np.ndarray, sx: np.ndarray, nc: np.ndarray, sc: np.ndarray) -> tuple[float, float, float]:
    """XBH mean minus the XBH-stratum-weighted control mean, over strata that have both groups."""
    ok = (nx > 0) & (nc > 0)
    x_mean = sx[ok].sum() / nx[ok].sum()
    c_mean = (nx[ok] * sc[ok] / nc[ok]).sum() / nx[ok].sum()
    return x_mean - c_mean, x_mean, c_mean


def weighted_did(xbh: pd.DataFrame, ctrl: pd.DataFrame, strata: list[str], value: str = "diff") -> dict:
    sx_code, sc_code, k = _stratum_codes(xbh, ctrl, strata)
    nx = np.bincount(sx_code, minlength=k).astype(float)
    nc = np.bincount(sc_code, minlength=k).astype(float)
    sx = np.bincount(sx_code, weights=xbh[value].to_numpy(), minlength=k)
    sc = np.bincount(sc_code, weights=ctrl[value].to_numpy(), minlength=k)
    net, x_mean, c_mean = _net_from_totals(nx, sx, nc, sc)
    ok = (nx > 0) & (nc > 0)
    return {"net": net, "xbh_mean": x_mean, "ctrl_mean": c_mean,
            "n_xbh": int(nx[ok].sum()), "n_xbh_all": len(xbh), "n_ctrl": int(nc[ok].sum())}


def cluster_bootstrap_did(
    xbh: pd.DataFrame, ctrl: pd.DataFrame, strata: list[str], value: str = "diff",
    cluster: str = "pitcher", n_boot: int = 1000, seed: int = 0,
) -> np.ndarray:
    """Bootstrap distribution of weighted_did's net, resampling `cluster` units with replacement."""
    sx_code, sc_code, k = _stratum_codes(xbh, ctrl, strata)
    clusters = pd.Index(pd.unique(pd.concat([xbh[cluster], ctrl[cluster]])))
    cx, cc = clusters.get_indexer(xbh[cluster]), clusters.get_indexer(ctrl[cluster])
    vx, vc = xbh[value].to_numpy(), ctrl[value].to_numpy()
    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    out = np.empty(n_boot)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters).astype(float)
        wx, wc = mult[cx], mult[cc]
        nx = np.bincount(sx_code, weights=wx, minlength=k)
        nc = np.bincount(sc_code, weights=wc, minlength=k)
        sx = np.bincount(sx_code, weights=wx * vx, minlength=k)
        sc = np.bincount(sc_code, weights=wc * vc, minlength=k)
        out[b] = _net_from_totals(nx, sx, nc, sc)[0]
    return out


def sample_1to1(xbh: pd.DataFrame, ctrl: pd.DataFrame, strata: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per stratum, keep min(n_xbh, n_ctrl) random rows of each group (the ladder's 1:1 design)."""
    rng = np.random.default_rng(seed)

    def draw(df: pd.DataFrame, quota: pd.Series) -> pd.DataFrame:
        d = df.assign(_r=rng.random(len(df)))
        d["_rank"] = d.groupby(strata, sort=False)["_r"].rank(method="first")
        q = d.join(quota.rename("_q"), on=strata)["_q"]
        return d[d["_rank"] <= q].drop(columns=["_r", "_rank"])

    quota = pd.concat([xbh.groupby(strata).size(), ctrl.groupby(strata).size()], axis=1).fillna(0).min(axis=1)
    return draw(xbh, quota), draw(ctrl, quota)


def bootstrap_ci(estimate: float, boot: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval."""
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def cluster_bootstrap_cells(
    events: pd.DataFrame, cell_cols: list[str], ref: tuple, value: str = "diff",
    cluster: str = "pitcher", n_boot: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """Each cell's mean minus the reference cell's mean, with pitcher-cluster bootstrap CIs."""
    cells = events.groupby(cell_cols, observed=True, sort=True).ngroup().to_numpy()
    labels = events.groupby(cell_cols, observed=True, sort=True).size().index
    ref_i = labels.get_loc(ref)
    k = len(labels)
    clusters = pd.Index(pd.unique(events[cluster]))
    ci = clusters.get_indexer(events[cluster])
    v = events[value].to_numpy()

    def diffs(w: np.ndarray) -> np.ndarray:
        n = np.bincount(cells, weights=w, minlength=k)
        s = np.bincount(cells, weights=w * v, minlength=k)
        m = s / n
        return m - m[ref_i]

    est = diffs(np.ones(len(events)))
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, k))
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)).astype(float)
        boot[b] = diffs(mult[ci])
    out = labels.to_frame(index=False)
    out["n"] = np.bincount(cells, minlength=k)
    out["vs_ref"] = est
    out["ci_low"], out["ci_high"] = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)
    return out
