"""
Data preprocessing pipeline — clean black-line flowchart
figures/08_preprocessing_pipeline.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

BLACK = "#000000"
WHITE = "#ffffff"
GRAY  = "#f0f0f0"   # light fill for file nodes only

# ── helpers ─────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, text, dashed=False, fill=WHITE):
    ls = (0, (4, 3)) if dashed else "solid"
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.015",
        facecolor=fill, edgecolor=BLACK, linewidth=1.2,
        linestyle=ls, zorder=3
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, fontsize=9, color=BLACK,
            ha="center", va="center", zorder=4,
            fontfamily="DejaVu Sans")

def arr(ax, x1, y1, x2, y2):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.0,
                        mutation_scale=10),
        zorder=2)

def line(ax, xs, ys):
    ax.plot(xs, ys, color=BLACK, lw=1.0, zorder=2)

def label(ax, x, y, text, fs=8, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fs, color=BLACK,
            ha="center", va="center", fontweight=fw)

def phase_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=8.5, color=BLACK,
            ha="center", va="center", fontweight="bold",
            fontstyle="italic")

def divider(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color=BLACK, lw=0.7,
            linestyle=(0, (6, 4)), zorder=1)

# ── canvas ───────────────────────────────────────────────────────────────────
W, H = 13.0, 8.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

ax.set_title("Data Preprocessing Pipeline",
             fontsize=13, fontweight="bold", color=BLACK, pad=8)

# ── phase dividers & labels ──────────────────────────────────────────────────
divider(ax, 4.3,  0.3, H - 0.5)
divider(ax, 8.6,  0.3, H - 0.5)

phase_label(ax, 2.15, H - 0.28, "(1)  Data Collection")
phase_label(ax, 6.45, H - 0.28, "(2)  Feature Extraction")
phase_label(ax, 10.8, H - 0.28, "(3)  ML Training")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Data Collection  (x center = 2.15)
# ════════════════════════════════════════════════════════════════════════════
BW, BH = 3.2, 0.52
PX = 2.15

Y = [7.5, 6.5, 5.5, 4.4, 3.2]

box(ax, PX, Y[0], BW, BH, "SMILES Input")
box(ax, PX, Y[1], BW, BH, "RDKit  ETKDGv3 + MMFF")
box(ax, PX, Y[2], BW, BH, "gaussian_writer.py  →  .com")
box(ax, PX, Y[3], BW, BH, "Gaussian 09   B3LYP / 6-31G(d)")
box(ax, PX, Y[4], BW * 0.75, BH, "mol.out", dashed=True, fill=GRAY)

arr(ax, PX, Y[0]-BH/2, PX, Y[1]+BH/2)
arr(ax, PX, Y[1]-BH/2, PX, Y[2]+BH/2)
arr(ax, PX, Y[2]-BH/2, PX, Y[3]+BH/2)
arr(ax, PX, Y[3]-BH/2, PX, Y[4]+BH/2)

# small note under Gaussian box
ax.text(PX, Y[3]-BH/2-0.13, "DFT geometry optimization",
        fontsize=7, color=BLACK, ha="center", va="top", fontstyle="italic")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Feature Extraction  (x center = 6.45)
# ════════════════════════════════════════════════════════════════════════════
MX = 6.45
BW2 = 3.6

# mol.out → two parse paths
OUT_X = PX + BW * 0.75 / 2   # right edge of mol.out box = 2.15 + 0.9 = 3.05
PARSE_Y1 = 6.3    # Input orientation
PARSE_Y2 = 5.1    # Standard orientation
TOPO_Y   = 3.9    # RDKit topology
FEAT_Y   = 2.7    # Bond Feature Extraction
CSV_Y    = 1.55   # bond_features.csv

# arrows from mol.out to parse boxes (via horizontal segment)
FORK_X = 4.55

# mol.out right → fork point
line(ax, [PX + BW * 0.75 / 2, FORK_X], [Y[4], Y[4]])
# fork → Input orientation
line(ax, [FORK_X, FORK_X], [Y[4], PARSE_Y1])
arr(ax, FORK_X, PARSE_Y1, MX - BW2/2, PARSE_Y1)
# fork → Standard orientation
line(ax, [FORK_X, FORK_X], [Y[4], PARSE_Y2])
arr(ax, FORK_X, PARSE_Y2, MX - BW2/2, PARSE_Y2)

box(ax, MX, PARSE_Y1, BW2, BH, "Input orientation   →   MMFF coords")
box(ax, MX, PARSE_Y2, BW2, BH, "Standard orientation  →   DFT coords")
box(ax, MX, TOPO_Y,   BW2, BH, "RDKit mol  (hybridization / ring info)")

# arrows into Bond Feature Extraction
arr(ax, MX, PARSE_Y1 - BH/2, MX, FEAT_Y + BH/2 + 0.5)
arr(ax, MX, PARSE_Y2 - BH/2, MX, FEAT_Y + BH/2 + 0.08)
arr(ax, MX, TOPO_Y   - BH/2, MX, FEAT_Y + BH/2)

box(ax, MX, FEAT_Y, BW2, BH, "Bond Feature Extraction   (per bond)")
arr(ax, MX, FEAT_Y - BH/2, MX, CSV_Y + BH/2)
box(ax, MX, CSV_Y, BW2 * 0.7, BH, "bond_features.csv", dashed=True, fill=GRAY)

# small note: dataset size
ax.text(MX, CSV_Y - BH/2 - 0.13, "529 bonds  /  49 molecules",
        fontsize=7, color=BLACK, ha="center", va="top", fontstyle="italic")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — ML Training  (x center = 10.8)
# ════════════════════════════════════════════════════════════════════════════
RX = 10.8
BW3 = 3.6

ML_Y = [7.2, 6.1, 5.0, 3.9, 2.7, 1.55]
ml_labels = [
    "Target :  dft_length  -  mmff_length",
    "OrdinalEncoder",
    "GradientBoostingRegressor",
    "5-Fold Cross Validation",
    "CV MAE = 0.0022  ±  0.0004  A",
    "models / *.joblib",
]

# csv → ML phase (horizontal then up)
CSV_RIGHT = MX + BW2 * 0.7 / 2
line(ax, [CSV_RIGHT, RX - BW3/2 - 0.05], [CSV_Y, CSV_Y])
arr(ax, RX - BW3/2 - 0.05, CSV_Y, RX - BW3/2, ML_Y[-1])

for i, (y, txt) in enumerate(zip(ML_Y, ml_labels)):
    dashed = (i == len(ML_Y) - 1)
    fill   = GRAY if dashed else WHITE
    box(ax, RX, y, BW3, BH, txt, dashed=dashed, fill=fill)
    if i < len(ML_Y) - 1:
        arr(ax, RX, y - BH/2, RX, ML_Y[i+1] + BH/2)

# note under GBR
ax.text(RX, ML_Y[2] - BH/2 - 0.13,
        "n_estimators=300   max_depth=4   lr=0.05",
        fontsize=7, color=BLACK, ha="center", va="top", fontstyle="italic")

# ── save ─────────────────────────────────────────────────────────────────────
out = Path("figures/08_preprocessing_pipeline.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved: {out}")
