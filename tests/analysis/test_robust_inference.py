import numpy as np
import pandas as pd

from src.analysis.robust_inference import (
    bootstrap_ci,
    cluster_bootstrap_cells,
    cluster_bootstrap_did,
    sample_1to1,
    weighted_did,
)


def _toy():
    # stratum A: 1 XBH (diff 0.5) vs 2 controls (0.1, 0.3); stratum B: 3 XBH (0.2) vs 1 control (0.0);
    # stratum C: 1 XBH with no control -> dropped
    xbh = pd.DataFrame({"s": ["A", "B", "B", "B", "C"], "diff": [0.5, 0.2, 0.2, 0.2, 9.0], "pitcher": [1, 2, 3, 4, 5]})
    ctrl = pd.DataFrame({"s": ["A", "A", "B"], "diff": [0.1, 0.3, 0.0], "pitcher": [1, 2, 3]})
    return xbh, ctrl


def test_weighted_did_weights_controls_by_xbh_stratum_counts():
    xbh, ctrl = _toy()
    r = weighted_did(xbh, ctrl, ["s"])
    assert r["n_xbh"] == 4 and r["n_xbh_all"] == 5 and r["n_ctrl"] == 3
    assert np.isclose(r["xbh_mean"], (0.5 + 0.6) / 4)
    assert np.isclose(r["ctrl_mean"], (1 * 0.2 + 3 * 0.0) / 4)
    assert np.isclose(r["net"], 0.275 - 0.05)


def test_cluster_bootstrap_is_centered_and_reproducible():
    rng = np.random.default_rng(1)
    n = 400
    xbh = pd.DataFrame({"s": rng.integers(0, 3, n), "pitcher": rng.integers(0, 50, n), "diff": rng.normal(0.15, 0.3, n)})
    ctrl = pd.DataFrame({"s": rng.integers(0, 3, n), "pitcher": rng.integers(0, 50, n), "diff": rng.normal(0.0, 0.3, n)})
    est = weighted_did(xbh, ctrl, ["s"])["net"]
    boot = cluster_bootstrap_did(xbh, ctrl, ["s"], n_boot=300, seed=7)
    lo, hi = bootstrap_ci(est, boot)
    assert lo < est < hi
    assert abs(boot.mean() - est) < 0.02
    assert np.array_equal(boot, cluster_bootstrap_did(xbh, ctrl, ["s"], n_boot=300, seed=7))


def test_sample_1to1_balances_each_stratum():
    xbh, ctrl = _toy()
    x, c = sample_1to1(xbh, ctrl, ["s"], seed=0)
    assert x.groupby("s").size().to_dict() == {"A": 1, "B": 1}
    assert c.groupby("s").size().to_dict() == {"A": 1, "B": 1}


def test_cluster_bootstrap_cells_reference_is_zero():
    ev = pd.DataFrame({"o": ["out"] * 4 + ["hr"] * 4, "diff": [0, 0, 0, 0, 1, 1, 1, 1], "pitcher": [1, 2, 3, 4] * 2})
    t = cluster_bootstrap_cells(ev, ["o"], "out", n_boot=50).set_index("o")
    assert t.loc["out", "vs_ref"] == 0 and np.isclose(t.loc["hr", "vs_ref"], 1)
    assert np.isclose(t.loc["hr", "ci_low"], 1) and np.isclose(t.loc["hr", "ci_high"], 1)
