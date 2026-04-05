"""
데이터 전처리 파이프라인 시각화
figures/08_preprocessing_pipeline.png 저장
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

# ── 색상 팔레트 ─────────────────────────────────────────────────────────────
C_INPUT   = "#4A90D9"   # 입력 (파랑)
C_RDKIT   = "#27AE60"   # RDKit (초록)
C_GAUSS   = "#E67E22"   # Gaussian (주황)
C_PARSE   = "#8E44AD"   # 파싱 (보라)
C_FEAT    = "#C0392B"   # Feature 추출 (빨강)
C_ML      = "#2C3E50"   # ML 학습 (다크)
C_DATA    = "#7F8C8D"   # 데이터 파일 (회색)
C_ARROW   = "#555555"
C_BG      = "#F8F9FA"

FONT_BOX  = dict(fontsize=9,  fontweight="bold", color="white",
                 ha="center", va="center")
FONT_SUB  = dict(fontsize=7.5, color="#EEEEEE", ha="center", va="center")
FONT_LABEL= dict(fontsize=7,  color="#333333", ha="center", va="center",
                 style="italic")

# ── 헬퍼 ────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, text, subtext="", radius=0.04):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    dy = 0.018 if subtext else 0
    ax.text(x, y + dy, text, **FONT_BOX, zorder=4)
    if subtext:
        ax.text(x, y - dy - 0.01, subtext, **FONT_SUB, zorder=4)

def file_box(ax, x, y, w, h, text):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.03",
        facecolor=C_DATA, edgecolor="#AAAAAA", linewidth=1, linestyle="--", zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=7.5, fontweight="bold", color="white",
            ha="center", va="center", zorder=4)

def arrow(ax, x1, y1, x2, y2, label="", color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.02, my, label, fontsize=6.5, color="#555555",
                ha="left", va="center", style="italic", zorder=5)

def bracket_arrow(ax, x1, y1, x2, y2):
    """꺾인 화살표 (수평 → 수직)"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.5, mutation_scale=12,
                                connectionstyle="arc3,rad=0"), zorder=2)

def section_bg(ax, x, y, w, h, color, alpha=0.07, label="", label_y=None):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=color, edgecolor=color, linewidth=0,
        alpha=alpha, zorder=0
    )
    ax.add_patch(rect)
    if label:
        lx = x + w/2
        ly = label_y if label_y else y + h + 0.02
        ax.text(lx, ly, label, fontsize=8, fontweight="bold",
                color=color, ha="center", va="bottom",
                alpha=min(1.0, alpha * 8), zorder=1)

# ── 메인 그림 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

ax.set_title("Data Preprocessing Pipeline\nSMILES → bond_features.csv → ML Model",
             fontsize=13, fontweight="bold", color="#2C3E50",
             pad=10, y=1.01)

# ════════════════════════════════════════════════════════════════════════════
# Section backgrounds
# ════════════════════════════════════════════════════════════════════════════
# Phase 1: Data Collection
section_bg(ax, 0.3, 5.6, 5.8, 3.1, C_GAUSS, alpha=0.08,
           label="(1) Data Collection  (collect_data.py)", label_y=8.78)
# Phase 2: Feature Extraction
section_bg(ax, 0.3, 1.3, 5.8, 4.0, C_PARSE, alpha=0.08,
           label="(2) Feature Extraction  (extract_features.py)", label_y=5.37)
# Phase 3: ML Training
section_bg(ax, 7.5, 1.3, 6.2, 7.5, C_ML, alpha=0.06,
           label="(3) ML Training  (train_model.py)", label_y=8.87)

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — 데이터 수집 (왼쪽 상단)
# ════════════════════════════════════════════════════════════════════════════
BW, BH = 2.2, 0.58

# 1-A: SMILES Input
box(ax, 1.9, 8.4, BW, BH, C_INPUT,
    "SMILES Input", "50 molecules")
arrow(ax, 1.9, 8.11, 1.9, 7.52)

# 1-B: RDKit ETKDGv3 + MMFF
box(ax, 1.9, 7.22, BW, BH, C_RDKIT,
    "RDKit  ETKDGv3 + MMFF", "3D conformer generation")
arrow(ax, 1.9, 6.93, 1.9, 6.34)

# 1-C: Gaussian .com
box(ax, 1.9, 6.04, BW, BH, C_GAUSS,
    "gaussian_writer.py", ".com file generation")
arrow(ax, 1.9, 5.75, 1.9, 5.18)

# 1-D: Gaussian 09
box(ax, 1.9, 4.88, 2.4, 0.64, C_GAUSS,
    "Gaussian 09  (B3LYP/6-31G(d))", "DFT geometry optimization")

# .out 파일
file_box(ax, 4.55, 6.5, 1.5, 0.45, "mol.out")
arrow(ax, 3.1, 4.88, 4.55, 4.88)
ax.annotate("", xy=(4.55, 6.27), xytext=(4.55, 4.88+0.32),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5, mutation_scale=12))

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Feature 추출 (왼쪽 하단)
# ════════════════════════════════════════════════════════════════════════════
# 2-A: Input orientation parsing
box(ax, 1.9, 3.95, BW, BH, C_PARSE,
    "Input orientation  parse", "1st block -> MMFF coords")
# .out -> parse (vertical arrow)
ax.annotate("", xy=(3.0, 3.95), xytext=(4.55, 6.27),
            arrowprops=dict(arrowstyle="-|>", color=C_PARSE,
                            lw=1.5, mutation_scale=12,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(3.35, 5.1, "mmff_coords", fontsize=6.5, color=C_PARSE,
        ha="center", style="italic")

# 2-B: Standard orientation parsing
box(ax, 1.9, 3.10, BW, BH, C_PARSE,
    "Standard orientation  parse", "last block -> DFT optimized coords")
ax.annotate("", xy=(0.85, 3.10), xytext=(4.55, 6.27),
            arrowprops=dict(arrowstyle="-|>", color=C_PARSE,
                            lw=1.5, mutation_scale=12,
                            connectionstyle="arc3,rad=0.35"))
ax.text(2.2, 4.65, "dft_coords", fontsize=6.5, color=C_PARSE,
        ha="center", style="italic")

# 2-C: RDKit topology (hybridization only)
box(ax, 1.9, 2.22, BW, BH, C_RDKIT,
    "RDKit mol  (topology)", "hybridization & ring info only\nETKDGv3 temp run")

# 2-D: Feature calculation
box(ax, 1.9, 1.42, 2.5, 0.62, C_FEAT,
    "Bond Feature Calculation", "one row per bond")

# 위 세 박스 → Feature 계산
arrow(ax, 1.9, 3.66, 1.9, 3.41)
arrow(ax, 1.9, 2.81, 1.9, 2.51)
arrow(ax, 1.9, 1.93, 1.9, 1.73)

# bond_features.csv
file_box(ax, 4.55, 1.42, 1.8, 0.50, "bond_features.csv\n(529 bonds, 49 mols)")
arrow(ax, 3.15, 1.42, 3.65, 1.42)

# ════════════════════════════════════════════════════════════════════════════
# Feature Table (center explanation)
# ════════════════════════════════════════════════════════════════════════════
feat_x, feat_y = 5.5, 4.2
ax.text(feat_x, feat_y + 0.55, "Bond Features  (per bond)",
        fontsize=8.5, fontweight="bold", color="#2C3E50",
        ha="center", va="center")

features = [
    ("elem1, elem2",        "element symbols  (alphabetical order)",        C_RDKIT),
    ("bond_order",          "1.0 / 1.5 / 2.0 / 3.0",                       C_RDKIT),
    ("hybridization_1/2",   "SP / SP2 / SP3",                               C_RDKIT),
    ("is_in_ring",          "in ring?  (0 or 1)",                           C_RDKIT),
    ("ring_size",           "smallest ring size  (0 = acyclic)",            C_RDKIT),
    ("mmff_length  (A)",    "bond length from Input orientation block",      C_PARSE),
    ("dft_length   (A)",    "bond length from Standard orientation  [target]", C_PARSE),
]

row_h = 0.36
table_top = feat_y + 0.28
for i, (name, desc, col) in enumerate(features):
    ry = table_top - i * row_h - row_h/2
    # 색 띠
    stripe = FancyBboxPatch(
        (feat_x - 2.1, ry - row_h/2 + 0.02), 4.2, row_h - 0.04,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=col, alpha=0.12, edgecolor=col, linewidth=0.5, zorder=1
    )
    ax.add_patch(stripe)
    ax.text(feat_x - 2.0, ry, name, fontsize=7.5, fontweight="bold",
            color=col, va="center", ha="left", zorder=2)
    ax.text(feat_x + 0.15, ry, desc, fontsize=7, color="#444444",
            va="center", ha="left", style="italic", zorder=2)

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — ML 학습 (오른쪽)
# ════════════════════════════════════════════════════════════════════════════
MLX = 10.6

# 3-A: Input CSV
file_box(ax, MLX, 8.1, 2.0, 0.50, "bond_features.csv")

# Route: csv -> ML pipeline (L-shaped arrow)
ax.annotate("", xy=(7.9, 1.42), xytext=(5.45, 1.42),
            arrowprops=dict(arrowstyle="-|>", color=C_ML, lw=2, mutation_scale=13))
ax.annotate("", xy=(MLX, 7.85), xytext=(MLX, 1.67),
            arrowprops=dict(arrowstyle="-|>", color=C_ML, lw=2, mutation_scale=13))
ax.plot([5.45, MLX], [1.42, 1.42], color=C_ML, lw=2, zorder=1)
ax.plot([MLX, MLX], [1.42, 1.67], color=C_ML, lw=2, zorder=1)

# 3-B: Delta target
box(ax, MLX, 7.22, 2.4, 0.58, C_FEAT,
    "Target (delta mode)", "dft_length - mmff_length  [A]")
arrow(ax, MLX, 7.85, MLX, 7.51)

# 3-C: OrdinalEncoder
box(ax, MLX, 6.35, 2.4, 0.58, C_ML,
    "OrdinalEncoder", "categorical features -> numeric")
arrow(ax, MLX, 6.93, MLX, 6.64)

# 3-D: GradientBoosting
box(ax, MLX, 5.45, 2.7, 0.62, C_ML,
    "GradientBoostingRegressor",
    "n=300  depth=4  lr=0.05  sub=0.8")
arrow(ax, MLX, 6.06, MLX, 5.76)

# 3-E: 5-Fold CV
box(ax, MLX, 4.5, 2.4, 0.58, C_ML,
    "5-Fold Cross Validation", "49 mols  (acetylene excluded)")
arrow(ax, MLX, 5.14, MLX, 4.79)

# 3-F: Performance result
perf_y = 3.5
ax.text(MLX, 4.19, "v", fontsize=11, ha="center", color=C_ML, fontweight="bold")
rect_perf = FancyBboxPatch(
    (MLX - 1.5, perf_y - 0.52), 3.0, 1.04,
    boxstyle="round,pad=0,rounding_size=0.04",
    facecolor="#1A252F", edgecolor="#27AE60", linewidth=2, zorder=3
)
ax.add_patch(rect_perf)
ax.text(MLX, perf_y + 0.2,  "CV MAE  =  0.0022 +/- 0.0004 A",
        fontsize=9.5, fontweight="bold", color="#2ECC71",
        ha="center", va="center", zorder=4)
ax.text(MLX, perf_y - 0.18, "Target < 0.005 A  achieved  [OK]",
        fontsize=8, color="#BDC3C7", ha="center", va="center", zorder=4)

# 3-G: Feature Importance summary
fi_y = 2.4
box(ax, MLX, fi_y, 2.6, 0.56, C_ML,
    "Feature Importance (top)",
    "mmff_length 80%  bond_order 6%  elem2 5%")
arrow(ax, MLX, perf_y - 0.52, MLX, fi_y + 0.28)

# Model artifact output
file_box(ax, MLX, 1.55, 2.0, 0.45, "models/*.joblib")
arrow(ax, MLX, fi_y - 0.28, MLX, 1.78)

# ════════════════════════════════════════════════════════════════════════════
# Legend
# ════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_INPUT, "Input Data"),
    (C_RDKIT, "RDKit Processing"),
    (C_GAUSS, "Gaussian 09"),
    (C_PARSE, ".out Parsing"),
    (C_FEAT,  "Feature / Target"),
    (C_ML,    "ML Training"),
    (C_DATA,  "File Artifact"),
]
handles = [mpatches.Patch(facecolor=c, label=l, edgecolor="white")
           for c, l in legend_items]
ax.legend(handles=handles, loc="lower right",
          bbox_to_anchor=(0.99, 0.01),
          fontsize=7.5, framealpha=0.9,
          ncol=1, edgecolor="#CCCCCC")

# ── 저장 ──────────────────────────────────────────────────────────────────
out_path = Path("figures/08_preprocessing_pipeline.png")
out_path.parent.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
plt.close()
print(f"저장 완료: {out_path}")
