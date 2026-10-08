"""Figures the final-report pipeline (`main.py`) redraws from its own
tables. They follow the figures in docs/최종정리.md but are regenerated, so
styling can differ from the report's.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from src.analysis.hit_quality import OUTCOME_ORDER
from src.analysis.rematch_execution import DELTA_LABELS

XBH_COLOR, CONTROL_COLOR, INK = "#D55E00", "#0072B2", "#333333"
KOREAN_FONTS = ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR")
PITCH_LABELS = {
    "FF": "포심 (FF)", "SI": "싱커 (SI)", "FC": "커터 (FC)", "SL": "슬라이더 (SL)", "ST": "스위퍼 (ST)",
    "CU": "커브 (CU)", "KC": "너클커브 (KC)", "SV": "슬러브 (SV)", "CH": "체인지업 (CH)", "FS": "스플리터 (FS)",
}
HAND_LABELS = {"R": "우투수", "L": "좌투수"}
FOREST_MEASURES = ["plate_x", "plate_z", "release_speed", "release_extension"]
OUTCOME_COLORS = {"아웃": "#009E73", "단타": "#56B4E9", "2·3루타": "#E69F00", "홈런": "#D55E00"}


def use_korean_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ladder(ladder: pd.DataFrame, path: Path) -> None:
    """Pure reduction per ladder step for the rematch (weighted control, pitcher-cluster CI)."""
    d = ladder[ladder["mode"] == "same_batter"]
    y = d["net"] * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.errorbar(d["step"], y, yerr=[y - d["ci_low"] * 100, d["ci_high"] * 100 - y], fmt="o", capsize=4, color=XBH_COLOR)
    ax.set_ylim(0, max(y) * 1.25)
    ax.set_ylabel("순수 감소 (%p, 95% CI)")
    ax.set_title("경기 상황을 맞춰도 순수 감소는 그대로", loc="left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_slots(by_slot: pd.DataFrame, path: Path) -> None:
    """Extra avoidance (XBH - control) toward each batter faced after the hit, then toward the rematch batter."""
    d = by_slot[by_slot["slot"] != "9"].reset_index(drop=True)  # the 9th-and-later slot has very few events
    x, y = np.arange(len(d)), d["net"] * 100
    err = np.vstack([y - d["ci_low"] * 100, d["ci_high"] * 100 - y])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(x[:-1], y[:-1], yerr=err[:, :-1], fmt="o-", capsize=4, color="#999999", label="중간 타자")
    ax.errorbar(x[-1:], y[-1:], yerr=err[:, -1:], fmt="o", capsize=4, color=XBH_COLOR, label="재대결 (장타 친 타자)")
    ax.set_xticks(x, [f"{s}번째" if s != "재대결" else "재대결\n(그 타자)" for s in d["slot"]])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("장타 뒤 타석 (중간 타자 k번째 → 재대결)")
    ax.set_ylabel("장타 - 대조군 회피 (%p, 95% CI)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_rematch_vs_control(table: pd.DataFrame, hand: str, path: Path) -> None:
    """Rows = pitch types, columns = measures; filled = after an XBH, hollow = after a field out."""
    sub = table[table["p_throws"] == hand]
    types = sub.drop_duplicates("hit_pitch_type").sort_values("n_xbh", ascending=False)["hit_pitch_type"].tolist()
    if not types:
        return
    fig, axes = plt.subplots(1, len(FOREST_MEASURES), figsize=(13.5, 0.62 * len(types) + 2.0), sharey=True, squeeze=False)
    y = np.arange(len(types), dtype=float)
    styles = (("xbh", XBH_COLOR, XBH_COLOR, -0.17, "장타 후 재대결"), ("control", CONTROL_COLOR, "white", 0.17, "범타 후 재대결 (대조군)"))
    for ax, measure in zip(axes[0], FOREST_MEASURES):
        m = sub[sub["measure"] == measure].set_index("hit_pitch_type").reindex(types)
        ax.axvline(0, color="#888888", lw=1)
        for name, color, face, offset, label in styles:
            ax.errorbar(m[name], y + offset, xerr=[m[name] - m[f"{name}_ci_low"], m[f"{name}_ci_high"] - m[name]],
                        fmt="o", ms=7, color=color, mfc=face, mew=1.6, lw=1.6, capsize=2.5, label=label)
        ax.set_title(DELTA_LABELS[measure][0])
        ax.grid(axis="x", color="#E5E5E5", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels([PITCH_LABELS.get(t, t) for t in types])
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.12), ncol=2)
    fig.suptitle(f"{HAND_LABELS[hand]}: 재대결 타석의 평소 대비 편차 (같은 구종, 평균과 95% CI)", fontsize=13, x=0.99, ha="right")
    fig.supxlabel("평소 대비 편차 (0 = 평소와 같음)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_hit_quality(cells: pd.DataFrame, path: Path) -> None:
    """Extra avoidance over a weak out, by result and xwOBA quartile."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, outcome in enumerate(OUTCOME_ORDER):
        o = cells[cells["outcome"] == outcome]
        x, y = np.arange(len(o)) + (i - 1.5) * 0.08, o["vs_ref"] * 100
        ax.errorbar(x, y, yerr=[y - o["ci_low"] * 100, o["ci_high"] * 100 - y], fmt="o-", capsize=3,
                    color=OUTCOME_COLORS[outcome], label=outcome)
    ax.set_xticks(range(4), ["Q1\n(약함)", "Q2", "Q3", "Q4\n(강함)"])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("xwOBA 분위")
    ax.set_ylabel("약한 아웃 대비 추가 회피 (%p, 95% CI)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_delta_by_usage(curve: pd.DataFrame, usage_by_group: pd.DataFrame, path: Path) -> None:
    """Avoided - reused selection-score gap along the hit type's usage (original specification),
    solid inside the range both groups cover, dotted outside it, with PA counts below."""
    inside = curve["in_support"].to_numpy()
    u, d, se = curve["usage"].to_numpy(), curve["delta"].to_numpy(), curve["se"].to_numpy()
    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08})
    ax.axhline(0, color="#888888", lw=1)
    ax.fill_between(u[inside], (d - 1.96 * se)[inside], (d + 1.96 * se)[inside], color="#BDBDBD", alpha=0.6, lw=0,
                    label="점별 95% 신뢰구간 (GBM 점수 고정)")
    ax.plot(u[inside], d[inside], color=INK, lw=2, label="차이: 공통 관측 구간")
    ax.plot(u[~inside], d[~inside], color=INK, lw=1.2, ls=":", label="외삽 구간 (결론에 사용하지 않음)")
    ax.set_ylabel("회피군 - 재사용군\n타석 내 평균 모델 선택점수 차이")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(axis="y", color="#E5E5E5", lw=0.7)
    edges = np.arange(0.0, u.max() + 1e-9, 0.05)
    for offset, group, color, label in ((-0.0105, 1, CONTROL_COLOR, "회피군"), (0.0105, 0, XBH_COLOR, "재사용군")):
        counts, _ = np.histogram(usage_by_group.loc[usage_by_group["avoided"] == group, "hit_usage_season"], bins=edges)
        ax_n.bar(edges[:-1] + 0.025 + offset, counts, width=0.019, color=color, label=label)
    ax_n.set_ylabel("타석 수\n(5%p 구간)")
    ax_n.set_xlabel("피장타 구종의 구사율")
    ax_n.legend(frameon=False, fontsize=9, loc="upper right")
    ax_n.set_xlim(0, u.max())
    for a in (ax, ax_n):
        a.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_shap_importance(importance: pd.DataFrame, path: Path, top: int = 11) -> None:
    """Mean |SHAP| of the strongest inputs; the top three are highlighted."""
    d = importance.head(top).iloc[::-1]
    colors = [XBH_COLOR if rank <= 3 else "#BBBBBB" for rank in d["rank"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(d["label"], d["mean_abs_shap"], color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xlim(0, d["mean_abs_shap"].max() * 1.15)
    ax.set_xlabel("평균 |SHAP| (회피 강도 예측에 대한 영향 크기)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)
