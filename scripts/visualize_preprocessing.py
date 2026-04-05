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
GAP = 0.35   # gap between box edge and arrow tip
XS  = [1.4, 3.9, 6.4, 8.9, 11.5]

labels = [
    ("SMILES Input",               False),
    ("RDKit  ETKDGv3 + MMFF",      False),
    ("gaussian_writer.py",         False),
    ("Gaussian 09  B3LYP/6-31G(d)", False),
    ("mol.out",                     True),
]

for x, (txt, dashed) in zip(XS, labels):
    bx(ax, x, CY, txt, w=BW, dashed=dashed)

for i in range(len(XS) - 1):
    arr(ax, XS[i] + BW/2, CY, XS[i+1] - BW/2, CY)

save("figures/08a_phase1_data_collection.png")


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Feature Extraction
#
#                        ┌─→ [Input orientation → MMFF coords] ─────┐
#  [mol.out] ────────────┤                                           │
#                        └─→ [Std orientation  → DFT coords ] ──────┤──→ [Bond Feature Extraction] ──→ [bond_features.csv]
#  [SMILES]  ──→ [RDKit mol (hybridization / ring)] ────────────────┘
#
# ════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
title(ax, "(2)  Feature Extraction")

BW2  = 2.9
HH   = 0.65   # box height

# X positions
X_out  = 1.3    # mol.out
X_smi  = 1.3    # SMILES  (below mol.out)
X_inp  = 4.7    # Input orientation
X_std  = 4.7    # Std orientation
X_rdk  = 4.7    # RDKit mol
X_feat = 8.8    # Bond Feature Extraction
X_csv  = 12.0   # bond_features.csv

# Y positions
Y_top = 3.2   # mol.out & Input orientation row
Y_mid = 2.25  # Standard orientation row
Y_bot = 1.3   # SMILES / RDKit mol row

# Source boxes
bx(ax, X_out, Y_top, "mol.out",  w=1.8, h=HH, dashed=True)
bx(ax, X_smi, Y_bot, "SMILES",   w=1.8, h=HH)

# Parse / topology boxes
bx(ax, X_inp, Y_top, "Input orientation  →  MMFF coords",  w=BW2, h=HH)
bx(ax, X_std, Y_mid, "Standard orientation  →  DFT coords", w=BW2, h=HH)
bx(ax, X_rdk, Y_bot, "RDKit mol  (hybridization / ring)",   w=BW2, h=HH)

# mol.out → Input orientation (straight right)
arr(ax, X_out + 1.8/2, Y_top, X_inp - BW2/2, Y_top)

# mol.out → Standard orientation (down then right)
FORK_X = X_out + 1.8/2 + 0.25
ln(ax, [X_out + 1.8/2, FORK_X], [Y_top, Y_top])
ln(ax, [FORK_X, FORK_X], [Y_top, Y_mid])
arr(ax, FORK_X, Y_mid, X_std - BW2/2, Y_mid)

# SMILES → RDKit mol (straight right)
arr(ax, X_smi + 1.8/2, Y_bot, X_rdk - BW2/2, Y_bot)

# Convergence: three parse boxes → Bond Feature Extraction
bx(ax, X_feat, Y_mid, "Bond Feature Extraction", w=2.6, h=HH)
MERGE_X = X_inp + BW2/2 + 0.2
for yi in [Y_top, Y_mid, Y_bot]:
    ln(ax, [X_inp + BW2/2, MERGE_X], [yi, yi])
ln(ax, [MERGE_X, MERGE_X], [Y_bot, Y_top])
arr(ax, MERGE_X, Y_mid, X_feat - 2.6/2, Y_mid)

# Bond Feature Extraction → csv
bx(ax, X_csv, Y_mid, "bond_features.csv", w=2.2, h=HH, dashed=True)
arr(ax, X_feat + 2.6/2, Y_mid, X_csv - 2.2/2, Y_mid)

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
