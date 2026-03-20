"""
absolute vs delta 타겟 모드 비교 분석

두 가지 타겟 모드로 모델을 학습하고 결과를 비교한다:
  - absolute: dft_length를 직접 예측
  - delta:    (dft_length - mmff_length) 보정량을 예측

출력:
  figures/05_target_mode_comparison.png  (4-panel 비교 그래프)
  figures/05_target_mode_comparison.csv  (결합 유형별 오차 분석)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from delta_chem.ml.train import train

FEATURES_CSV = "data/features/bond_features.csv"
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

print("=" * 60)
print("  Target Mode Comparison: absolute vs delta")
print("=" * 60)

# ── 두 모드 학습 ─────────────────────────────────────────────────
print("\n[1/2] absolute 모드 학습...")
res_abs = train(
    features_csv=FEATURES_CSV,
    model_out="models/bond_length_corrector_absolute.joblib",
    exclude=["acetylene"],
    target_mode="absolute",
)

print("\n[2/2] delta 모드 학습...")
res_del = train(
    features_csv=FEATURES_CSV,
    model_out="models/bond_length_corrector_delta.joblib",
    exclude=["acetylene"],
    target_mode="delta",
)

# ── 결과 요약 ────────────────────────────────────────────────────
abs_mae = res_abs["cv_mae"].mean()
abs_std = res_abs["cv_mae"].std()
del_mae = res_del["cv_mae"].mean()
del_std = res_del["cv_mae"].std()

print("\n" + "=" * 60)
print(f"{'':30s} {'absolute':>12s}  {'delta':>12s}")
print("=" * 60)
print(f"{'5-Fold CV MAE (Å)':<30s} {abs_mae:>12.5f}  {del_mae:>12.5f}")
print(f"{'CV MAE std (Å)':<30s} {abs_std:>12.5f}  {del_std:>12.5f}")
print(f"{'개선율':<30s} {'':>12s}  {(1 - del_mae/abs_mae)*100:>+11.1f}%")
print("=" * 60)

# ── 결합 유형별 오차 분석 ────────────────────────────────────────
df = res_abs["df"].copy()

# absolute 모드: y_pred = dft_length 예측
df["pred_abs"] = res_abs["y_pred"]
df["err_abs"]  = (df["pred_abs"] - df["dft_length"]).abs()

# delta 모드: y_pred = delta 예측 → dft_length 공간으로 변환
df["pred_del_dft"] = df["mmff_length"] + res_del["y_pred"]
df["err_del"]      = (df["pred_del_dft"] - df["dft_length"]).abs()

# 결합 유형 레이블 생성
def bond_label(row):
    bo_map = {1.0: "s", 1.5: "ar", 2.0: "d", 3.0: "t"}
    bo = bo_map.get(row["bond_order"], "?")
    return f"{row['elem1']}-{row['elem2']}({bo})"

df["bond_type"] = df.apply(bond_label, axis=1)

bt_stats = df.groupby("bond_type").agg(
    n=("err_abs", "count"),
    mae_abs=("err_abs", "mean"),
    mae_del=("err_del", "mean"),
).reset_index()
bt_stats["improvement"] = (1 - bt_stats["mae_del"] / bt_stats["mae_abs"]) * 100
bt_stats = bt_stats.sort_values("n", ascending=False)

print("\n결합 유형별 MAE 비교 (훈련셋 전체):")
print(f"{'결합 유형':<14s} {'n':>5s}  {'MAE_abs':>10s}  {'MAE_del':>10s}  {'개선율':>8s}")
print("-" * 54)
for _, r in bt_stats.iterrows():
    print(f"{r['bond_type']:<14s} {int(r['n']):>5d}  "
          f"{r['mae_abs']*1000:>8.3f}mÅ  {r['mae_del']*1000:>8.3f}mÅ  {r['improvement']:>+7.1f}%")

# CSV 저장
csv_path = FIGDIR / "05_target_mode_comparison.csv"
bt_stats.to_csv(csv_path, index=False)
print(f"\nCSV 저장: {csv_path}")

# ── 시각화 ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Target Mode Comparison: Absolute vs Delta",
             fontsize=14, fontweight="bold")

colors = {"absolute": "#5B9BD5", "delta": "#ED7D31"}

# Panel 1: CV MAE per fold
ax = axes[0, 0]
folds = np.arange(1, 6)
ax.plot(folds, res_abs["cv_mae"] * 1000, "o-", color=colors["absolute"],
        label=f"Absolute  (mean={abs_mae*1000:.2f} mÅ)")
ax.plot(folds, res_del["cv_mae"] * 1000, "s--", color=colors["delta"],
        label=f"Delta     (mean={del_mae*1000:.2f} mÅ)")
ax.set_xlabel("Fold")
ax.set_ylabel("MAE (mÅ)")
ax.set_title("5-Fold Cross-Validation MAE")
ax.set_xticks(folds)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Parity plot — absolute
ax2 = axes[0, 1]
y_true_abs = df["dft_length"].values
ax2.scatter(y_true_abs, df["pred_abs"].values,
            s=8, alpha=0.4, color=colors["absolute"])
lo, hi = y_true_abs.min() - 0.01, y_true_abs.max() + 0.01
ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax2.set_xlabel("DFT length (Å)")
ax2.set_ylabel("Predicted (Å)")
ax2.set_title(f"Parity — Absolute  MAE={abs_mae*1000:.2f} mÅ")
ax2.set_aspect("equal")

# Panel 3: Parity plot — delta
ax3 = axes[1, 0]
ax3.scatter(y_true_abs, df["pred_del_dft"].values,
            s=8, alpha=0.4, color=colors["delta"])
ax3.plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax3.set_xlabel("DFT length (Å)")
ax3.set_ylabel("Predicted (Å)")
ax3.set_title(f"Parity — Delta  MAE={del_mae*1000:.2f} mÅ")
ax3.set_aspect("equal")

# Panel 4: 결합 유형별 MAE 비교 막대그래프
ax4 = axes[1, 1]
top_bt = bt_stats.head(10)  # 상위 10개 결합 유형만
x = np.arange(len(top_bt))
w = 0.38
b1 = ax4.bar(x - w/2, top_bt["mae_abs"] * 1000, w,
             label="Absolute", color=colors["absolute"], edgecolor="white")
b2 = ax4.bar(x + w/2, top_bt["mae_del"] * 1000, w,
             label="Delta", color=colors["delta"], edgecolor="white")
ax4.set_xticks(x)
ax4.set_xticklabels(top_bt["bond_type"], rotation=35, ha="right", fontsize=8)
ax4.set_ylabel("Train MAE (mÅ)")
ax4.set_title("Per Bond-Type Train MAE")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
png_path = FIGDIR / "05_target_mode_comparison.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"그래프 저장: {png_path}")
print("\n완료.")
