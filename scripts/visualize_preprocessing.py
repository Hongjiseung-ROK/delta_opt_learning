"""
Pipeline visualization — one image per phase
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

def draw(nodes, edges, title, out_path,
         figw=5, figh=6, node_w=3.2, node_h=0.55):
    """
    nodes : [(cx, cy, text, dashed), ...]
    edges : [(x1,y1, x2,y2), ...]   straight arrows
    """
    fig, ax = plt.subplots(figsize=(figw, figh))
    ax.set_xlim(0, figw); ax.set_ylim(0, figh)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.set_title(title, fontsize=11, fontweight="bold", color=BLACK, pad=6)

    for (cx, cy, text, dashed) in nodes:
        ls   = (0, (5, 3)) if dashed else "solid"
        fill = GRAY if dashed else WHITE
        rect = mpatches.FancyBboxPatch(
            (cx - node_w/2, cy - node_h/2), node_w, node_h,
            boxstyle="round,pad=0,rounding_size=0.02",
            facecolor=fill, edgecolor=BLACK, linewidth=1.1,
            linestyle=ls, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, text, fontsize=9, color=BLACK,
                ha="center", va="center", zorder=4)

    for (x1, y1, x2, y2) in edges:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK,
                                    lw=1.0, mutation_scale=10), zorder=2)

    Path(out_path).parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


# ── Phase 1 : Data Collection ────────────────────────────────────────────────
cx = 2.5
ys = [5.2, 4.2, 3.2, 2.2, 1.2]
h  = 0.55
gap = h / 2

draw(
    nodes=[
        (cx, ys[0], "SMILES Input",                  False),
        (cx, ys[1], "RDKit  ETKDGv3 + MMFF",         False),
        (cx, ys[2], "gaussian_writer.py  →  .com",   False),
        (cx, ys[3], "Gaussian 09  B3LYP/6-31G(d)",   False),
        (cx, ys[4], "mol.out",                        True),
    ],
    edges=[
        (cx, ys[0]-gap, cx, ys[1]+gap),
        (cx, ys[1]-gap, cx, ys[2]+gap),
        (cx, ys[2]-gap, cx, ys[3]+gap),
        (cx, ys[3]-gap, cx, ys[4]+gap),
    ],
    title="(1)  Data Collection",
    out_path="figures/08a_phase1_data_collection.png",
    figw=5, figh=6.2,
)


# ── Phase 2 : Feature Extraction ─────────────────────────────────────────────
#
#         mol.out
#        /       \
# Input orient.   Std orient.   RDKit topology
#        \            |         /
#         Bond Feature Extraction
#               bond_features.csv
#
figw, figh = 6.5, 6.5
cx  = figw / 2
nw  = 2.8
nh  = 0.52
gap = nh / 2

# x positions for three source boxes
x1, x2, x3 = 1.3, 3.25, 5.2
out_y   = 5.6
src_y   = 4.1
feat_y  = 2.6
csv_y   = 1.3

fig, ax = plt.subplots(figsize=(figw, figh))
ax.set_xlim(0, figw); ax.set_ylim(0, figh)
ax.axis("off")
fig.patch.set_facecolor(WHITE)
ax.set_title("(2)  Feature Extraction", fontsize=11,
             fontweight="bold", color=BLACK, pad=6)

def bx(cx, cy, txt, dashed=False, w=nw, h=nh):
    ls   = (0, (5, 3)) if dashed else "solid"
    fill = GRAY if dashed else WHITE
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=fill, edgecolor=BLACK, linewidth=1.1,
        linestyle=ls, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, txt, fontsize=8.5, color=BLACK,
            ha="center", va="center", zorder=4)

def ar(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=BLACK,
                                lw=1.0, mutation_scale=10), zorder=2)

def ln(xs, ys):
    ax.plot(xs, ys, color=BLACK, lw=1.0, zorder=2)

# mol.out (top center)
bx(cx, out_y, "mol.out", dashed=True, w=2.0)

# three source boxes
bx(x1, src_y, "Input orientation\n→  MMFF coords",   w=2.4)
bx(x2, src_y, "Standard orientation\n→  DFT coords", w=2.4)
bx(x3, src_y, "RDKit mol\n(hybridization / ring)",    w=2.4)

# mol.out → Input orient  (left branch)
ln([cx, x1], [out_y - nh/2, out_y - nh/2])
ar(x1, out_y - nh/2, x1, src_y + nh/2 + 0.15)

# mol.out → Std orient  (center branch)
ar(cx, out_y - nh/2, x2, src_y + nh/2 + 0.15)

# RDKit: independent source, just arrow down from outside
ax.annotate("", xy=(x3, src_y + nh/2 + 0.15), xytext=(x3, out_y - nh/2),
            arrowprops=dict(arrowstyle="-|>", color=BLACK,
                            lw=1.0, mutation_scale=10,
                            linestyle="dashed"), zorder=2)
ax.text(x3, (src_y + nh/2 + out_y - nh/2) / 2 + 0.1,
        "(SMILES)", fontsize=7, color=BLACK, ha="center",
        va="center", fontstyle="italic")

# three boxes → Bond Feature Extraction
bx(cx, feat_y, "Bond Feature Extraction", w=3.4)
for xi in [x1, x2, x3]:
    ln([xi, xi], [src_y - nh/2, feat_y + nh/2 + 0.35])
    ar(xi, feat_y + nh/2 + 0.35, cx - 0.01 if xi < cx else cx + 0.01, feat_y + nh/2)

# Bond Features → csv
ar(cx, feat_y - nh/2, cx, csv_y + nh/2)
bx(cx, csv_y, "bond_features.csv", dashed=True, w=2.8)

plt.savefig("figures/08b_phase2_feature_extraction.png",
            dpi=180, bbox_inches="tight", facecolor=WHITE, edgecolor="none")
plt.close()
print("Saved: figures/08b_phase2_feature_extraction.png")


# ── Phase 3 : ML Training ────────────────────────────────────────────────────
cx = 2.5
ys3 = [5.5, 4.5, 3.5, 2.5, 1.5, 0.6]
h   = 0.55
gap = h / 2

draw(
    nodes=[
        (cx, ys3[0], "bond_features.csv",              True),
        (cx, ys3[1], "Target :  dft_length - mmff_length", False),
        (cx, ys3[2], "OrdinalEncoder  +  GradientBoosting", False),
        (cx, ys3[3], "5-Fold Cross Validation",        False),
        (cx, ys3[4], "CV MAE = 0.0022 ± 0.0004  Å",   False),
        (cx, ys3[5], "models / *.joblib",               True),
    ],
    edges=[
        (cx, ys3[0]-gap, cx, ys3[1]+gap),
        (cx, ys3[1]-gap, cx, ys3[2]+gap),
        (cx, ys3[2]-gap, cx, ys3[3]+gap),
        (cx, ys3[3]-gap, cx, ys3[4]+gap),
        (cx, ys3[4]-gap, cx, ys3[5]+gap),
    ],
    title="(3)  ML Training",
    out_path="figures/08c_phase3_ml_training.png",
    figw=5, figh=6.5,
)
