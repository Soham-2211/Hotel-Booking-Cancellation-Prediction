"""Static report figures (PNG) with one consistent, colour-blind-checked style."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
SERIES = {"City": BLUE, "Resort": ORANGE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cleaning_impact(log_df, path):
    d = log_df[log_df["rows"] > 0].sort_values("rows")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ORANGE if a == "drop row" else BLUE for a in d["action"]]
    ax.barh(d["rule"], d["rows"], color=colors, height=0.6)
    ax.set_xscale("log")
    for y, (v, a) in enumerate(zip(d["rows"], d["action"])):
        ax.text(v * 1.08, y, f"{v:,}  ({a})", va="center", fontsize=8, color=INK2)
    ax.set_xlim(right=d["rows"].max() * 400)
    ax.set_title("Records affected by each cleaning rule (orange = rows dropped)")
    ax.set_xlabel("rows (cells for whitespace padding), log scale")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def cancel_by_lead_time(df, path):
    bins = [-1, 7, 30, 60, 90, 180, 365, 1000]
    labels = ["0-7", "8-30", "31-60", "61-90", "91-180", "181-365", "365+"]
    d = df.assign(bucket=np.asarray(np.digitize(df["lead_time"], bins[1:-1], right=True)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for hotel, color in SERIES.items():
        r = d[d["hotel"] == hotel].groupby("bucket")["is_canceled"].mean().reindex(range(len(labels)))
        ax.plot(labels, r.values, marker="o", markersize=6, color=color)
        ax.text(len(labels) - 0.9, r.values[-1], hotel, color=INK, va="center", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(-0.3, len(labels) - 0.2)
    ax.set_title("Cancellation rate rises steeply with lead time")
    ax.set_xlabel("days between booking and arrival")
    ax.set_ylabel("cancellation rate")
    _save(fig, path)


def cancel_over_time(df, path):
    m = df.groupby([df["arrival_date"].dt.to_period("M"), "hotel"])["is_canceled"].mean().unstack()
    x = m.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    last = m.iloc[-1]
    hi = last.idxmax()
    for hotel, color in SERIES.items():
        ax.plot(x, m[hotel], color=color)
        dy = 0.025 if hotel == hi else -0.025
        ax.text(x[-1] + np.timedelta64(12, "D"), m[hotel].iloc[-1] + dy, hotel, va="center",
                fontweight="bold")
    for t, lab in [("2017-01-01", "valid"), ("2017-05-01", "test")]:
        ax.axvline(np.datetime64(t), color=INK2, lw=1, ls="--")
        ax.text(np.datetime64(t) + np.timedelta64(6, "D"), 0.07, lab, color=INK2, fontsize=9)
    ax.text(np.datetime64("2015-08-01"), 0.07, "train", color=INK2, fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(right=x[-1] + np.timedelta64(75, "D"))
    ax.set_title("Monthly cancellation rate and the temporal split")
    _save(fig, path)


def split_comparison(rows, path):
    """rows: list of (label, roc_auc, kind) where kind in {'honest','optimistic','leaky'}."""
    color = {"honest": BLUE, "optimistic": MUTED, "leaky": ORANGE}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    ax.barh(labels, vals, color=[color[r[2]] for r in rows][::-1], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.8, 1.02)
    ax.set_xlabel("ROC AUC")
    ax.set_title("Same model, different evaluation: how leakage inflates scores")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def roc_pr(y, preds: dict, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = [MUTED, BLUE]
    for (name, p), c in zip(preds.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=c, label=name)
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=c, label=name)
    a1.plot([0, 1], [0, 1], color=GRID, lw=1)
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (test season)")
    a2.axhline(np.mean(y), color=GRID, lw=1)
    a2.set(xlabel="recall", ylabel="precision", title="Precision-recall (test season)")
    a1.legend(loc="lower right")
    _save(fig, path)


def calibration(y, p, path):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.plot(mean, frac, marker="o", markersize=6, color=BLUE)
    ax.set(xlabel="predicted probability", ylabel="observed cancellation rate",
           title="Calibration (test season)", xlim=(0, 1), ylim=(0, 1))
    _save(fig, path)


def importance(imp_df, path, top=15):
    d = imp_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color=BLUE,
            height=0.6, error_kw={"ecolor": INK2, "lw": 1})
    ax.set_xlabel("drop in ROC AUC when feature is shuffled")
    ax.set_title(f"Permutation importance, top {top} (test season)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def value_curve(grid, vals, chosen, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, vals / 1e3, color=BLUE)
    ax.axhline(0, color=INK2, lw=1)
    i = int(np.argmin(np.abs(grid - chosen)))
    ax.plot([chosen], [vals[i] / 1e3], marker="o", markersize=8, color=ORANGE)
    ax.annotate(f"threshold {chosen:.2f}\n(chosen on validation)", (chosen, vals[i] / 1e3),
                textcoords="offset points", xytext=(12, -30), fontsize=9, color=INK)
    ax.set(xlabel="overbook when P(cancel) >= threshold", ylabel="net value (EUR thousands)",
           title="Overbooking value vs decision threshold (test season)")
    _save(fig, path)
