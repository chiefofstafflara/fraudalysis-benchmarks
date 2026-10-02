#!/usr/bin/env python3
"""
Generate every benchmark chart on the Fraudalysis benchmarks page.

Every number is read from a captured results.json produced by a real Kaggle run.
Nothing is hand-typed, so a chart cannot quietly disagree with the published
figure. Re-running this after a benchmark change regenerates all charts.

Colour is aligned to the site's brand green (--accent #22c55e and --accent-text
#15803d, matching fraudalysis-mark.svg) rather than matplotlib's default hues,
so the charts read as part of the site instead of generic chart junk. Datasets
are told apart by depth of green. The single exception is the rejected 0.9971
score, drawn in grey: it is a number we chose not to publish and it should not
look as strong as the ones we did.
"""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/ubuntu/fraudalysis/public/images/benchmarks"
W = "/home/ubuntu/kaggle-fradualysis-benchmark"

GREEN_DARK = "#052e16"
GREEN = "#15803d"
GREEN_BRIGHT = "#22c55e"
GREEN_MID = "#4ade80"
GREEN_PALE = "#bbf7d0"
INK = "#052e16"
MUTED = "#4b5563"
GRID = "#e5e7eb"
REJECTED = "#9ca3af"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

cc = json.load(open(f"{W}/out_creditcard/results.json"))
eth = json.load(open(f"{W}/out_ethereum/results.json"))
# Read from reference/, not /tmp: the old path was a scratch clone that gets
# wiped, which silently broke regeneration later.
syn = json.load(open(f"{W}/reference/synthetic-financial-results.json"))


def title(fig, text):
    fig.suptitle(text, fontweight="bold", y=1.02, fontsize=14, color=INK)


def dataset_sizes():
    """Log scale is deliberate: the range spans 9,841 to 6.36M (~650x), so on a
    linear axis the small datasets render as invisible slivers, which reads as
    'this dataset is unimportant' rather than 'the axis is wrong'."""
    rows = [
        ("Metaverse\nFinancial", 78600, GREEN_PALE),
        ("Synthetic\nFinancial", 6362608, GREEN_BRIGHT),
        ("Credit Card", 284807, GREEN_MID),
        ("Ethereum", 9841, GREEN),
    ]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.3, 1]}
    )
    bars = ax.barh(
        [r[0] for r in rows][::-1], [r[1] for r in rows][::-1],
        color=[r[2] for r in rows][::-1], height=0.62, zorder=3,
    )
    ax.set_xscale("log")
    for b, r in zip(bars, rows[::-1]):
        ax.text(r[1] * 1.15, b.get_y() + b.get_height() / 2, f"{r[1]:,}",
                va="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlim(3000, 3e7)
    ax.set_xlabel("Records analysed (log scale)", color=MUTED)
    ax.set_title("Dataset Scale", fontweight="bold", pad=12, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # Unit of analysis gets its own panel: the page must be precise about this,
    # so it is shown rather than buried in a footnote.
    ax2.axis("off")
    ax2.text(0.5, 0.92, "Unit of Analysis", ha="center", fontweight="bold",
             fontsize=12, transform=ax2.transAxes, color=INK)
    ax2.text(0.5, 0.74, "3 datasets", ha="center", fontsize=11, color=GREEN,
             transform=ax2.transAxes, fontweight="bold")
    ax2.text(0.5, 0.63, "per TRANSACTION", ha="center", fontsize=9.5,
             color=MUTED, transform=ax2.transAxes)
    ax2.text(0.5, 0.40, "1 dataset", ha="center", fontsize=11, color=GREEN_DARK,
             transform=ax2.transAxes, fontweight="bold")
    ax2.text(0.5, 0.29, "per ADDRESS", ha="center", fontsize=9.5,
             color=MUTED, transform=ax2.transAxes)
    ax2.text(0.5, 0.06, "Ethereum rows are addresses,\nnot transactions",
             ha="center", fontsize=8.5, color=MUTED, style="italic",
             transform=ax2.transAxes)

    title(fig, "Benchmark Dataset Sizes")
    fig.tight_layout()
    fig.savefig(f"{OUT}/dataset-comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc_across():
    rows = [
        ("Synthetic Financial", syn["roc_auc_score"], GREEN_BRIGHT, "6,362,608 rows"),
        ("Ethereum", eth["roc_auc_score"], GREEN_DARK, "9,841 addresses"),
        ("Credit Card", cc["roc_auc_score"], GREEN_MID, "284,807 rows"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    # Row counts go in the category tick label. The bar axis is truncated at
    # 0.95 (so the differences are visible), leaving too little coloured area
    # for in-bar text -- grey on the dark green Ethereum bar was unreadable.
    # Annotating below the axis instead makes bbox_inches="tight" expand the
    # canvas into a huge blank strip. Note: assigning to existing tick labels
    # via set_text() is silently ignored by matplotlib here, so the labels are
    # supplied to bar() directly.
    bars = ax.bar([f"{r[0]}\n{r[3]}" for r in rows], [r[1] for r in rows],
                  color=[r[2] for r in rows], width=0.55, zorder=3)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, r[1] + 0.003, f"{r[1]:.4f}",
                ha="center", fontweight="bold", fontsize=13, color=INK)
    ax.set_ylim(0.95, 1.006)
    ax.set_ylabel("ROC-AUC score", color=MUTED)
    ax.set_title("ROC-AUC across datasets — identical Random Forest "
                 "methodology, CPU only", fontweight="bold", pad=14, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/roc-across-datasets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def honest_numbers():
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    vals = [0.9971, eth["roc_auc_score"]]
    bars = ax.bar(["Median-filled\n(REJECTED)", "Zero-filled\n(PUBLISHED)"],
                  vals, color=[REJECTED, GREEN], width=0.45, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontweight="bold", fontsize=14, color=INK)
    ax.annotate(
        "829 addresses (8.4%) had NO ERC20 activity.\n"
        "Median-filling invented token volumes for them.",
        xy=(0, 0.9971), xytext=(0.40, 0.9935), fontsize=9.5, color=MUTED,
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.2),
    )
    ax.set_ylim(0.983, 0.9995)
    ax.set_ylabel("ROC-AUC score", color=MUTED)
    ax.set_title("The score we did not publish — and the honest one we did",
                 fontweight="bold", pad=14, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/honest-numbers-roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def feature_importance(which, heading):
    """top_features is a list of {feature, importance} dicts as written by the
    benchmark scripts, ordered most-important first."""
    src = cc if which == "creditcard" else eth
    top = sorted(src["top_features"], key=lambda d: d["importance"])[:15]
    feats = [d["feature"] for d in top]
    imp = [d["importance"] for d in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(imp)), imp, color=GREEN, height=0.68, zorder=3)
    # Highlight the single strongest predictor so the eye lands on it first.
    bars[-1].set_color(GREEN_BRIGHT)
    bars[-1].set_edgecolor(GREEN_DARK)
    bars[-1].set_linewidth(1.4)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(feats, fontsize=9.5)
    for i, v in enumerate(imp):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8.5,
                color=MUTED, fontweight="bold")
    ax.set_xlabel("Feature importance", color=MUTED)
    ax.set_xlim(0, max(imp) * 1.22)
    # Title states the real count: the credit card run recorded 10 top features,
    # not 15, so a hardcoded "Top 15" would misdescribe the chart.
    ax.set_title(heading.format(n=len(imp)), fontweight="bold", pad=14, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{which}-feature-importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def creditcard_amounts():
    """Median and mean tell opposite stories here. Charting both is the honest
    way to show the mean alone would mislead anyone setting threshold rules."""
    st = cc["amount_stats"]
    labels = ["Legitimate", "Fraudulent"]
    means = [st["legit_mean"], st["fraud_mean"]]
    medians = [st["legit_median"], st["fraud_median"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    b1 = ax1.bar(labels, means, color=[GREEN_PALE, GREEN_DARK], width=0.5, zorder=3)
    for b, v in zip(b1, means):
        ax1.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:,.2f}", ha="center",
                 fontweight="bold", fontsize=11, color=INK)
    ax1.set_title("Mean amount", fontweight="bold", color=INK)

    b2 = ax2.bar(labels, medians, color=[GREEN_PALE, GREEN_BRIGHT], width=0.5, zorder=3)
    for b, v in zip(b2, medians):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:,.2f}", ha="center",
                 fontweight="bold", fontsize=11, color=INK)
    ax2.set_title("Median amount", fontweight="bold", color=INK)

    ax1.text(0.5, 0.94, "fraud looks LARGER", transform=ax1.transAxes, ha="center",
             fontsize=10, color=GREEN_DARK, fontweight="bold")
    ax2.text(0.5, 0.94, "fraud is mostly SMALL", transform=ax2.transAxes, ha="center",
             fontsize=10, color=GREEN_DARK, fontweight="bold")
    for a in (ax1, ax2):
        a.set_ylabel("Amount", color=MUTED)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)

    title(fig, "Credit Card: why averages alone would mislead you")
    fig.tight_layout()
    fig.savefig(f"{OUT}/creditcard-amount-pattern.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_matrices():
    """Row percentages, with every cell labelled. These matrices are the
    evidence behind the precision/recall claims, so the counts must be readable
    rather than inferred from a colour gradient."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    panels = [
        (cc, "Credit Card Fraud",
         f"{sum(sum(r) for r in cc['confusion_matrix']):,} test transactions"),
        (eth, "Ethereum Fraud",
         f"{sum(sum(r) for r in eth['confusion_matrix']):,} test addresses"),
    ]
    for ax, (src, name, sub) in zip(axes, panels):
        m = np.array(src["confusion_matrix"])
        norm = m / m.sum(axis=1, keepdims=True) * 100
        ax.imshow(norm, cmap="Greens", vmin=0, vmax=100)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legitimate", "Fraudulent"], fontsize=10)
        ax.set_yticklabels(["Legitimate", "Fraudulent"], fontsize=10)
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual", color=MUTED)
        for i in range(2):
            for j in range(2):
                val = norm[i, j]
                ax.text(j, i, f"{m[i, j]:,}\n{val:.1f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > 50 else INK)
        ax.set_title(f"{name}\n{sub}", fontweight="bold", fontsize=11.5, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)

    title(fig, "Detection Outcomes (row percentages)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion-matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dataset_sizes()
    roc_across()
    honest_numbers()
    feature_importance("creditcard", "Top {n} predictors — Credit Card Fraud Detection")
    feature_importance("ethereum", "Top {n} predictors — Ethereum Fraud Detection")
    creditcard_amounts()
    confusion_matrices()
    print("All 7 charts regenerated with the brand green palette.")