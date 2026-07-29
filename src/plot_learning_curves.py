"""Draw the per-class learning curves produced by ``learning_curves.py``.

Small multiples, one panel per class, because the reader's job is per class: the
whole point of the figure is that the nine transitions are in different regimes,
and a single overlaid axis would hide that behind nine crossing lines. Each panel
carries exactly two series -- the out-of-fold score and the training score -- so
the gap between them is readable at a glance:

    both curves high and together   -> learned, and generalising
    train high, OOF flat and low    -> variance / memorised label noise
    both low and flat               -> the features do not separate this class

Colour does identity only inside a panel (two series, blue/orange from the
validated categorical order, all-pairs safe); class identity is carried by the
panel title, which is why nine classes need no nine hues.

The x-axis is that class's **own** training count, not the global fraction. A
per-class learning curve asked in global fractions is unreadable when the class
prevalences span 46 to 2,532 plots -- the rare classes' curves would all be
squeezed into the first tick.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Validated categorical slots 1 and 2 (light mode, all-pairs: CVD dE 24.7,
# normal-vision dE 33.6, both >= 3:1 on the surface).
OOF_COLOR = "#2a78d6"
TRAIN_COLOR = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"


def _panel(ax, sub, xcol):
    """One class: OOF and train mean +/- 1 sd over seeds, against training size."""
    for split, color, label in (("oof", OOF_COLOR, "Out-of-fold"),
                                ("train", TRAIN_COLOR, "Train")):
        d = sub[sub["split"] == split]
        if d.empty:
            continue
        g = d.groupby(xcol)["f1"].agg(["mean", "std", "size"]).reset_index()
        x, m = g[xcol].to_numpy(), g["mean"].to_numpy()
        s = np.nan_to_num(g["std"].to_numpy(), nan=0.0)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=color, linewidth=2.0, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=label,
                zorder=3, clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def _figure(df, level, xcol, title, subtitle, path, ncols=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df["level"] == level]
    real = sorted(c for c in d["cls"].unique() if c not in ("MACRO", "CHANGE"))
    panels = real + ["CHANGE", "MACRO"]
    nrows = int(np.ceil(len(panels) / ncols))
    # A fixed header strip in inches, so the title block never has to negotiate
    # with the panels for space (which is what made them collide).
    header = 1.15
    height = 2.9 * nrows + header
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, height),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, cls in zip(axes, panels):
        sub = d[d["cls"] == cls]
        # The aggregate reads have no per-class training count, so they are
        # drawn against the total instead -- stated in the panel label.
        col = "n_train" if cls in ("MACRO", "CHANGE") else xcol
        _panel(ax, sub, col)
        ax.set_facecolor(SURFACE)
        support = int(sub[sub["split"] == "oof"]["support"].max())
        if cls == "CHANGE":
            name, note = "Any change (binary gate)", "all plots"
        elif cls == "MACRO":
            name, note = "Macro-F1 (unweighted)", "all plots"
        else:
            name, note = cls, f"{support:,} plots"
        # Two titles on one line: the name in ink at left, the support count in
        # muted ink at right. Neither can overrun the other.
        ax.set_title(name, fontsize=10, color=INK, pad=7, loc="left")
        ax.set_title(note, fontsize=8, color=MUTED, pad=7, loc="right")
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    # x-label on the bottom-most *visible* panel of each column only.
    bottom = {i % ncols: i for i in range(len(panels))}
    for i, ax in enumerate(axes[:len(panels)]):
        if i % ncols == 0:
            ax.set_ylabel("F1", fontsize=9, color=MUTED)
        if bottom[i % ncols] == i:
            ax.set_xlabel("training plots for this class (log)",
                          fontsize=9, color=MUTED)

    fig.tight_layout(rect=(0, 0, 1, 1 - header / height))
    fig.text(0.008, 1 - 0.30 / height, title, fontsize=14, color=INK,
             ha="left", va="top", weight="medium")
    fig.text(0.008, 1 - 0.66 / height, subtitle, fontsize=9, color=MUTED,
             ha="left", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, fontsize=9,
               labelcolor=MUTED, loc="upper right",
               bbox_to_anchor=(0.995, 1 - 0.22 / height))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def draw(csv: Path, out_dir: Path) -> list[Path]:
    df = pd.read_csv(csv)
    n_seeds = df["seed"].nunique()
    sub = (f"Deployed recipe s2off_centre_m3s3_bf, gate-off read, spatially "
           f"blocked 5-fold CV, {n_seeds} seeds. Band is +/-1 sd over seeds.")
    made = []
    made.append(_figure(
        df, "coarse3", "n_train_cls",
        "Learning curves by transition class (coarse3, the 9-class read)",
        sub, out_dir / "learning_curves_coarse3.png", ncols=4))
    made.append(_figure(
        df, "merged2", "n_train_cls",
        "Learning curves by transition class (merged2, the deployed read)",
        sub, out_dir / "learning_curves_merged2.png", ncols=3))
    for p in made:
        print(f"-> {p}")
    return made


if __name__ == "__main__":
    import sys
    from project_paths import project_data_dir

    out = project_data_dir("analysis_results")
    draw(Path(sys.argv[1]) if len(sys.argv) > 1 else out / "learning_curves.csv",
         out)
