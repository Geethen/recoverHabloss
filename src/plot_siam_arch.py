"""Draw the architecture of the deployment-shaped siamese recipe.

`siam_s2off_state_pre` (`infer_s2.fit_models`, sections N and P7 of
`docs/research/SIAMESE_RESEARCH.md`) is the deployed recipe
`s2off_centre_m3s3_bf` with its AlphaEarth tower replaced by a shared endpoint
encoder, that encoder pretrained on single-date land-cover states. This figure
exists because neither change is visible in the kwargs -- `aef_siam=True` and
`siam_state_pretrain=30` -- and both are entirely structural underneath.

**Three** things the figure has to carry, because they are the facts a reader
gets wrong about this model. Each is a lane, and only the middle one is served:

* **The Sentinel-2 lane is training-only.** Drawn as a dashed enclosure below
  the served path rather than as a peer tower, because `deploy="aef_only"`
  forces its gate to zero and `probs_aef_only_matrix` never builds it.
* **The GLanCE lane is training-only AND runs first.** Drawn above, as a
  separate phase, because what crosses from it into the model is *weights* and
  nothing else -- no tensor, no feature, no posterior. Everything the project
  tried before P7 fed state labels in as a term alongside the transition loss
  and came back flat; this is a phase, and the arrow has to say so or the figure
  shows the thing that did not work.
* **The encoder is one module applied twice**, not two towers. Drawn once, with
  both endpoint blocks entering it -- and it is the *same* module phase 1
  trained, which is the whole mechanism.

Colour does path identity only, from the validated categorical order the
learning-curve figure uses: slot 1 for the served AlphaEarth path, slot 2 for
the privileged Sentinel-2 path, slot 3 for the state-pretraining phase.
Everything else is ink on the surface.

    $P src/plot_siam_arch.py
"""
from __future__ import annotations

from pathlib import Path

# Validated categorical slots 1, 2 and 3, as in plot_learning_curves.py.
AEF = "#2a78d6"
S2 = "#eb6834"
STATE = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"

TITLE = "siam_s2off_state_pre — state-pretrained shared endpoint encoder, Sentinel-2 as privileged information"
SUBTITLE = ("The deployed recipe s2off_centre_m3s3_bf with its AlphaEarth tower swapped for a siamese "
            "encoder (aef_siam=True), that encoder pretrained on single-date land-cover states "
            "(siam_state_pretrain=30).\nServed AlphaEarth-only: the detail gate is forced off, so neither "
            "training-only lane is built at inference and serving cost is unchanged.")
NOTE = ("Read merged2 for WHETHER change happened and coarse3 for WHAT KIND: arg-max does not commute "
        "with the group sum, so the two disagree on ~15% of change pixels.")
FOOTER = ("src/infer_s2.py:784 (recipe) · src/model_zoo.py:3074 (_pretrain_state) · :4026 (encode_single, "
          "the single-date entry point) · :3846 (_SiameseTrunk) · :1774 and :2141 (the hierarchy)")


def _tint(color, alpha):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return tuple(1 - alpha * (1 - c) for c in (r, g, b))


def _box(ax, x, y, w, h, title=None, lines=(), color=AEF, alpha=0.07,
         dashed=False, body_size=8.2, lw=1.3, spacing=2.3):
    """A rounded block: title in ink, body lines in muted ink, left-aligned.

    `spacing` is in data units, and one data unit is 7.2pt here, so the default
    is a ~16pt leading on 8pt text. Anything looser and the blocks stop reading
    as single objects.
    """
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=lw, edgecolor=color, facecolor=_tint(color, alpha),
        linestyle="--" if dashed else "-", zorder=2))
    cy = y + h - 3.6
    if title:
        ax.text(x + 2.6, cy, title, fontsize=9.5, color=INK, va="center",
                ha="left", zorder=3)
        cy -= 4.4
    for line in lines:
        ax.text(x + 2.6, cy, line, fontsize=body_size, color=MUTED,
                va="center", ha="left", zorder=3, family="DejaVu Sans")
        cy -= spacing


def _chip(ax, x, y, w, h, label, sub=None, color=AEF):
    """A small centred node: a tensor, not a computation."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
        linewidth=1.3, edgecolor=color, facecolor=_tint(color, 0.13), zorder=2))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h / 2 + 1.5, label, fontsize=9, color=INK,
                ha="center", va="center", zorder=3)
        ax.text(cx, y + h / 2 - 2.0, sub, fontsize=7.6, color=MUTED,
                ha="center", va="center", zorder=3)
    else:
        ax.text(cx, y + h / 2, label, fontsize=9, color=INK,
                ha="center", va="center", zorder=3)


def _arrow(ax, p0, p1, color=MUTED, rad=0.0, ls="-"):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
        color=color, linestyle=ls, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}", zorder=1.6))


def draw(out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One data unit is 10 px at this scale in both axes, so the two lanes added
    # for phase 1 cost exactly their height and nothing reflows.
    fig = plt.figure(figsize=(16.8, 13.4), facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 168)
    ax.set_ylim(-6, 128)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    ax.text(2, 123.5, TITLE, fontsize=15, color=INK, ha="left", va="top")
    ax.text(2, 119.4, SUBTITLE, fontsize=9, color=MUTED, ha="left", va="top",
            linespacing=1.5)

    # ---- phase 1: state pretraining on single-date labels -----------------
    # Above the served path and in its own enclosure, because it is not a branch
    # of the forward graph -- it runs to completion first and leaves only
    # weights. Drawn with the same dashed idiom as the Sentinel-2 lane so the
    # two read as one category (training-only), and in a different hue so they
    # do not read as one mechanism.
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (1, 84), 121, 30, boxstyle="round,pad=0,rounding_size=2.0",
        linewidth=1.2, edgecolor=STATE, facecolor=_tint(STATE, 0.04),
        linestyle=(0, (5, 4)), zorder=1))
    ax.text(3.5, 111.4, "PHASE 1 — PRETRAINING.  Runs first, once, and to completion.  The state head is "
                        "then discarded and the auxiliary term is OFF for the whole of phase 2; no GLanCE "
                        "is read at inference.",
            fontsize=8.2, color=STATE, ha="left", va="center")

    _box(ax, 4, 87, 38, 21, "GLanCE 2018  —  single-date states", [
        "13,118 units, 83 blocks — the same blocks as the plots",
        "strict subset; Segment_Type is null on 95.7% of the asset",
        "harmonised to {Nature, Cropland, Artificial}",
        "legend cleared against RECOVER:",
        "macro-F1 0.733 vs a 0.740 self-floor",
    ], STATE, body_size=7.5)

    _chip(ax, 47, 93.5, 15, 8, "x", "ONE date, 64 cols", STATE)
    _arrow(ax, (42, 97), (47, 97.5), STATE)

    _box(ax, 67, 87, 30, 21, "f  via  encode_single(x)", [
        "the SAME shared encoder as phase 2 —",
        "f never needed the pair, only the head",
        "above it did.  This is why a single-date",
        "label can enter this model at all: the flat",
        "trunk eats [2018 | 2024 | diff] and has",
        "no single-date input.",
    ], STATE, body_size=7.5)
    _arrow(ax, (62, 97.5), (67, 97), STATE)

    _box(ax, 101, 87, 19, 21, "state head  g", [
        "Linear 128 → 3",
        "cross-entropy",
        "",
        "30 epochs · AdamW",
        "BN stats frozen",
        "DISCARDED after",
    ], STATE, body_size=7.5)
    _arrow(ax, (97, 97), (101, 97), STATE)

    # The one arrow in the figure that carries parameters rather than
    # activations, drawn heavier and labelled as such -- a reader who takes it
    # for a feature path has been shown N14, which is the thing that failed.
    # Its caption sits in the gap between the lanes and to the LEFT of the
    # arrow, because everything right of x=95 at this height is the head block.
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        (76, 84), (63, 74.6), arrowstyle="-|>", mutation_scale=15,
        linewidth=2.4, color=STATE, shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=0.12", zorder=1.6))
    ax.text(4, 80.4, "initialises f — WEIGHTS ONLY.",
            fontsize=8.4, color=STATE, ha="left", va="center")
    ax.text(4, 77.6, "No tensor, no feature and no posterior crosses this boundary;\n"
                     "phase 2 starts from these weights and never sees the pool again.",
            fontsize=7.8, color=MUTED, ha="left", va="center", linespacing=1.45)

    # ---- served path: AlphaEarth, siamese ---------------------------------
    _box(ax, 2, 48, 22, 18, "AlphaEarth", [
        "192 columns",
        "64 bands × {2018, 2024, diff}",
        "any column order — the tower",
        "regroups them itself",
    ], AEF, body_size=7.8)

    _chip(ax, 28, 62, 14, 8, "x₁₈", "64 cols", AEF)
    _chip(ax, 28, 50, 14, 8, "x₂₄", "64 cols", AEF)
    _chip(ax, 28, 37, 14, 8, "diff", "64 cols", AEF)
    _arrow(ax, (24, 60), (28, 66), AEF, rad=0.12)
    _arrow(ax, (24, 57), (28, 54), AEF, rad=0.0)
    _arrow(ax, (24, 53), (28, 42), AEF, rad=-0.12)

    _box(ax, 48, 48, 27, 26, "shared encoder  f  — applied twice", [
        "Linear    64 → 512 · BN · GELU · Drop 0.4",
        "Linear  512 → 256 · BN · GELU · Drop 0.4",
        "Linear  256 → 128        ← linear on purpose",
        "",
        "both dates in ONE stacked call, so BN pools",
        "2018 and 2024 and cannot re-centre the",
        "between-year shift — which is the signal",
        "",
        "PHASE 2 starts from phase 1's weights",
    ], AEF, body_size=7.8)
    _arrow(ax, (42, 66), (48, 68), AEF)
    _arrow(ax, (42, 54), (48, 58), AEF)
    ax.text(45, 61.5, "same\nweights", fontsize=7.2, color=MUTED,
            ha="center", va="center", linespacing=1.3)

    _chip(ax, 78, 64, 12, 8, "z₁₈", "128-d", AEF)
    _chip(ax, 78, 52, 12, 8, "z₂₄", "128-d", AEF)
    _arrow(ax, (75, 68), (78, 68), AEF)
    _arrow(ax, (75, 58), (78, 56), AEF)

    _box(ax, 95, 40, 32, 36, "head block of the siamese trunk", [
        "concat   z₁₈ · z₂₄ · (z₂₄ − z₁₈) · |z₂₄ − z₁₈| · cos",
        "         513 wide,  ⊕ the 64 diff cols  →  577",
        "",
        "mixer    Linear  577 → 512 · BN · GELU · Drop",
        "         Linear  512 → 256 · GELU",
        "",
        "cos(z₁₈, z₂₄) is supervised directly, w 0.3,",
        "margin 0.3: stable pairs pulled together,",
        "change pairs pushed apart. Carried by the",
        "stable majority, so it costs the rare classes",
        "nothing.",
    ], AEF, body_size=7.9)
    _arrow(ax, (90, 68), (95, 66), AEF)
    _arrow(ax, (90, 56), (95, 60), AEF)
    _arrow(ax, (42, 40), (95, 44), AEF, rad=0.03)
    ax.text(68, 36.6, "the diff block bypasses the encoder — it is not a per-year measurement",
            fontsize=7.6, color=MUTED, ha="center", va="center")

    _chip(ax, 133, 63, 20, 9, "r_aef", "256-d", AEF)
    _arrow(ax, (127, 60), (133, 67), AEF, rad=0.08)

    # ---- privileged path: Sentinel-2, training only -----------------------
    ax.add_patch(FancyBboxPatch(
        (1, 5), 121, 29, boxstyle="round,pad=0,rounding_size=2.0",
        linewidth=1.2, edgecolor=S2, facecolor=_tint(S2, 0.04),
        linestyle=(0, (5, 4)), zorder=1))
    ax.text(3.5, 7.8, "PHASE 2, PRIVILEGED INFORMATION — training only.  deploy=\"aef_only\" forces this "
                      "gate to zero; probs_aef_only_matrix never builds this lane, and no "
                      "Sentinel-2 is fetched at inference.",
            fontsize=8.2, color=S2, ha="left", va="center")

    _box(ax, 4, 13, 40, 18, "Sentinel-2 detail  —  78 columns", [
        "7 channels  (blue, green, red, NIR, NDVI, NDWI, brightness)",
        "× {centre, 3×3 mean, 3×3 sd}                        = 21",
        "+ built fraction at 5 radii                         = 26 / year",
        "× {2018, 2024, diff}                                = 78",
    ], S2, body_size=7.8)

    _box(ax, 50, 15, 42, 16, "detail tower  (MLP)", [
        "Linear     78 → 1024 · BN · GELU · Drop 0.7",
        "Linear  1024 →  512 · BN · GELU · Drop 0.7",
        "Linear   512 →  256 · GELU",
    ], S2)
    _arrow(ax, (44, 22), (50, 23), S2)

    _chip(ax, 99, 18, 18, 9, "r_s2", "gated to 0 at serve", S2)
    _arrow(ax, (92, 23), (99, 23), S2)

    # ---- fusion -----------------------------------------------------------
    _box(ax, 130, 45, 36, 17, "gated mean fusion", [
        "(g_a · r_aef  +  g_t · r_s2)  /  max(g_a + g_t, 1)",
        "",
        "serve:  g_t = 0  ⇒  fused = r_aef",
    ], INK, alpha=0.05, body_size=7.8)
    _arrow(ax, (143, 65), (143, 62), AEF)
    _arrow(ax, (117, 25), (130.5, 48), S2, rad=0.3)

    # ---- the hierarchy ----------------------------------------------------
    # Drawn as a chain hanging off ONE softmax rather than as three outputs of
    # a head, because that is the fact the structure turns on: there is no gate
    # head and no merged head, and both coarse levels are 0/1 group sums of the
    # same nine probabilities.
    _box(ax, 130, 1, 36, 41, "hierarchy — three reads of ONE softmax", [
        "one head:   Linear 256 → 9 logits · softmax",
        "no gate head, no merged head, no extra parameters",
        "loss = w_f·L(p_fine) + w_m·L(p_merged) + w_g·L(p_gate)",
    ], INK, alpha=0.05, body_size=7.7)
    _arrow(ax, (143, 45), (143, 42), MUTED)

    _chip(ax, 132, 21, 20, 5.5, "p_fine", "9 — coarse3, what kind", INK)
    _arrow(ax, (142, 21), (142, 18), MUTED)
    ax.text(145.5, 19.5, "@ M\n9×4", fontsize=7.4, color=MUTED,
            ha="left", va="center", linespacing=1.35)

    _chip(ax, 132, 12.5, 20, 5.5, "p_merged", "4 — merged2, whether", INK)
    _arrow(ax, (142, 12.5), (142, 9.5), MUTED)
    ax.text(145.5, 11.0, "@ G\n4×2", fontsize=7.4, color=MUTED,
            ha="left", va="center", linespacing=1.35)

    _chip(ax, 132, 4, 20, 5.5, "p_gate", "2 — change / stable", INK)

    ax.text(2, -1.4, NOTE, fontsize=8.4, color=INK, ha="left", va="center")
    ax.text(2, -4.0, FOOTER, fontsize=7.6, color=MUTED, ha="left", va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for ext in ("png", "svg"):
        path = out_dir / f"siam_s2off_state_pre_architecture.{ext}"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        made.append(path)
    plt.close(fig)
    return made


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/analysis_results")
    for p in draw(out):
        print(f"-> {p}")
