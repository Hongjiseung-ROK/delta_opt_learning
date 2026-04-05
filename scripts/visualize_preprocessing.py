"""
Pipeline visualization — horizontal layout, right-pointing arrows (PPT optimized)
figures/08a_phase1_data_collection.png
figures/08b_phase2_feature_extraction.png
figures/08c_phase3_ml_training.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BLACK = "#000000"
WHITE = "#ffffff"
GRAY  = "#eeeeee"

FW, FH = 13.0, 4.5   # landscape 16:9 friendly


def make_fig():
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    return fig, ax


def bx(ax, cx, cy, txt, w=2.2, h=0.7, dashed=False):
    ls   = (0, (5, 3)) if dashed else "solid"
    fill = GRAY if dashed else WHITE
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=fill, edgecolor=BLACK, linewidth=1.2,
        linestyle=ls, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, txt, fontsize=9, color=BLACK,
            ha="center", va="center", zorder=4)


def arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=BLACK,
                                lw=1.1, mutation_scale=11), zorder=2)


def ln(ax, xs, ys):
    ax.plot(xs, ys, color=BLACK, lw=1.1, zorder=2)


def title(ax, txt):
    ax.set_title(txt, fontsize=12, fontweight="bold", color=BLACK, pad=7)


def save(path):
    Path(path).parent.mkdir(exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.close()
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — Data Collection   (linear left → right)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
title(ax, "(1)  Data Collection")

CY  = FH / 2
BW  = 2.2
BH  = 0.85   # taller boxes to accommodate two-line text
XS  = [1.4, 3.9, 6.4, 8.9, 11.5]

labels = [
    ("SMILES Input",                False),
    ("RDKit\nETKDGv3 + MMFF",       False),
    ("gaussian_writer.py",          False),
    ("Gaussian 09\nB3LYP/6-31G(d)", False),
    ("mol.out",                      True),
]

for x, (txt, dashed) in zip(XS, labels):
    bx(ax, x, CY, txt, w=BW, h=BH, dashed=dashed)

for i in range(len(XS) - 1):
    arr(ax, XS[i] + BW/2, CY, XS[i+1] - BW/2, CY)

save("figures/08a_phase1_data_collection.png")


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Feature Extraction
#
#  All arrowheads point RIGHT only.
#  Vertical segments are plain lines (no arrowhead).
#
#  [mol.out] ─F─────────────→ [Input orientation → MMFF coords] ─┐M
#              │                                                    │ │
#              └─────────────→ [Std orientation  → DFT coords ] ──┤ ├──→ [Bond Feature] ──→ [csv]
#                                                                   │ │
#  [SMILES]  ──────────────→ [RDKit mol (hybridization / ring)] ──┘ │
#
# F = fork point (vertical connector, no arrowhead)
# M = merge point (vertical collector, no arrowhead), one arrow goes right
# ════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
title(ax, "(2)  Feature Extraction")

SRC_W  = 1.8    # mol.out / SMILES box width
PARSE_W = 3.0   # parse box width
FEAT_W  = 2.4   # Bond Feature Extraction width
CSV_W   = 2.0   # csv width
HH      = 0.72  # box height

# X centers
X_src   = 1.2
FORK_X  = X_src + SRC_W/2 + 0.35          # fork junction
X_parse = 5.4                               # parse boxes center
MERGE_X = X_parse + PARSE_W/2 + 0.3        # merge junction
X_feat  = MERGE_X + 0.3 + FEAT_W/2 + 0.5
X_csv   = X_feat + FEAT_W/2 + 0.4 + CSV_W/2

# Y rows
Y_top = 3.4
Y_mid = 2.25
Y_bot = 1.1

# ── source boxes ────────────────────────────────────────────
bx(ax, X_src, Y_top, "mol.out", w=SRC_W, h=HH, dashed=True)
bx(ax, X_src, Y_bot, "SMILES",  w=SRC_W, h=HH)

# ── parse / topology boxes ───────────────────────────────────
bx(ax, X_parse, Y_top, "Input orientation  →  MMFF coords",   w=PARSE_W, h=HH)
bx(ax, X_parse, Y_mid, "Standard orientation  →  DFT coords", w=PARSE_W, h=HH)
bx(ax, X_parse, Y_bot, "RDKit mol  (hybridization / ring)",   w=PARSE_W, h=HH)

# ── mol.out fork (no downward arrowhead) ─────────────────────
#    horizontal line: mol.out right → FORK_X  (Y_top)
ln(ax, [X_src + SRC_W/2, FORK_X], [Y_top, Y_top])
#    vertical connector: FORK_X from Y_top down to Y_mid
ln(ax, [FORK_X, FORK_X], [Y_mid, Y_top])
#    right arrows from FORK_X to each parse box
arr(ax, FORK_X, Y_top, X_parse - PARSE_W/2, Y_top)   # → Input orientation
arr(ax, FORK_X, Y_mid, X_parse - PARSE_W/2, Y_mid)   # → Std orientation

# ── SMILES → RDKit mol ───────────────────────────────────────
arr(ax, X_src + SRC_W/2, Y_bot, X_parse - PARSE_W/2, Y_bot)

# ── merge (no arrowhead on collector lines) ───────────────────
#    horizontal lines: each parse box right → MERGE_X
for yi in [Y_top, Y_mid, Y_bot]:
    ln(ax, [X_parse + PARSE_W/2, MERGE_X], [yi, yi])
#    vertical collector at MERGE_X
ln(ax, [MERGE_X, MERGE_X], [Y_bot, Y_top])
#    single right arrow from MERGE_X mid → Bond Feature Extraction
bx(ax, X_feat, Y_mid, "Bond Feature Extraction", w=FEAT_W, h=HH)
arr(ax, MERGE_X, Y_mid, X_feat - FEAT_W/2, Y_mid)

# ── Bond Feature Extraction → csv ────────────────────────────
bx(ax, X_csv, Y_mid, "bond_features.csv", w=CSV_W, h=HH, dashed=True)
arr(ax, X_feat + FEAT_W/2, Y_mid, X_csv - CSV_W/2, Y_mid)

save("figures/08b_phase2_feature_extraction.png")


# ════════════════════════════════════════════════════════════════════════════
# Phase 3 — ML Training   (linear left → right)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
title(ax, "(3)  ML Training")

CY  = FH / 2
BW3 = 2.2
XS3 = [1.3, 3.7, 6.3, 8.8, 11.3]

labels3 = [
    ("bond_features.csv",                  True),
    ("Target :\ndft_length - mmff_length", False),
    ("OrdinalEncoder\n+ GradientBoosting", False),
    ("5-Fold\nCross Validation",           False),
    ("CV MAE\n0.0022 ± 0.0004  Å",        False),
]

for x, (txt, dashed) in zip(XS3, labels3):
    bx(ax, x, CY, txt, w=BW3, h=0.85, dashed=dashed)

for i in range(len(XS3) - 1):
    arr(ax, XS3[i] + BW3/2, CY, XS3[i+1] - BW3/2, CY)

save("figures/08c_phase3_ml_training.png")
