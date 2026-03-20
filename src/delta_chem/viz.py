"""공통 시각화 유틸리티."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless (GUI 없이 파일 저장)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FIGURES_DIR = Path("figures")
STYLE = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}


def savefig(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  그래프 저장: {path}")
    return path


# ── Feature 추출 단계 ────────────────────────────────────────────────

def plot_mmff_vs_dft(df, out_name: str = "01_mmff_vs_dft.png") -> None:
    """MMFF 결합 길이 vs DFT 결합 길이 산점도 (결합 유형별 색상)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))

        bond_types = (df["elem1"] + "-" + df["elem2"] +
                      df["bond_order"].apply(lambda x: {1.0:"(s)",1.5:"(ar)",2.0:"(d)",3.0:"(t)"}.get(x,""))).values
        unique = sorted(set(bond_types))
        cmap = plt.cm.get_cmap("tab10", len(unique))
        color_map = {b: cmap(i) for i, b in enumerate(unique)}

        for bt in unique:
            mask = bond_types == bt
            ax.scatter(df["mmff_length"][mask], df["dft_length"][mask],
                       label=bt, color=color_map[bt], s=30, alpha=0.8)

        lims = [min(df["mmff_length"].min(), df["dft_length"].min()) - 0.02,
                max(df["mmff_length"].max(), df["dft_length"].max()) + 0.02]
        ax.plot(lims, lims, "k--", lw=0.8, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("MMFF bond length (Å)")
        ax.set_ylabel("B3LYP/6-31G(d) bond length (Å)")
        ax.set_title("MMFF vs DFT Bond Lengths")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.set_aspect("equal")
        fig.tight_layout()
        savefig(fig, out_name)


def plot_correction_distribution(df, out_name: str = "02_correction_dist.png") -> None:
    """(DFT - MMFF) 보정값 분포 히스토그램 (결합 유형별)."""
    df = df.copy()
    df["delta"] = df["dft_length"] - df["mmff_length"]
    bond_types = (df["elem1"] + "-" + df["elem2"] +
                  df["bond_order"].apply(lambda x: {1.0:"(s)",1.5:"(ar)",2.0:"(d)",3.0:"(t)"}.get(x,""))).values
    unique = sorted(set(bond_types))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        for bt in unique:
            mask = bond_types == bt
            vals = df["delta"].values[mask]
            ax.hist(vals, bins=15, alpha=0.6, label=f"{bt} (n={mask.sum()})")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("DFT − MMFF (Å)")
        ax.set_ylabel("Count")
        ax.set_title("Bond Length Correction Distribution")
        ax.legend(fontsize=7)
        fig.tight_layout()
        savefig(fig, out_name)


def save_correction_stats_csv(df, out_name: str = "02_correction_stats.csv") -> None:
    """결합 유형별 평균 보정값 CSV."""
    import pandas as pd
    df = df.copy()
    df["delta"] = df["dft_length"] - df["mmff_length"]
    df["bond_type"] = (df["elem1"] + "-" + df["elem2"] +
                       df["bond_order"].apply(lambda x: {1.0:"(s)",1.5:"(ar)",2.0:"(d)",3.0:"(t)"}.get(x,"")))
    stats = df.groupby("bond_type")["delta"].agg(["count","mean","std","min","max"])
    stats.columns = ["n", "mean_delta_A", "std_A", "min_A", "max_A"]
    stats = stats.round(5)
    path = FIGURES_DIR / out_name
    FIGURES_DIR.mkdir(exist_ok=True)
    stats.to_csv(path)
    print(f"  통계 CSV 저장: {path}")


# ── 학습 단계 ────────────────────────────────────────────────────────

def plot_feature_importance(feature_names, importances,
                             out_name: str = "03_feature_importance.png") -> None:
    idx = np.argsort(importances)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([feature_names[i] for i in idx], importances[idx], color="steelblue")
        ax.set_xlabel("Feature Importance")
        ax.set_title("Gradient Boosting – Feature Importance")
        fig.tight_layout()
        savefig(fig, out_name)


def plot_parity(y_true, y_pred, mae, out_name: str = "04_parity.png") -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, s=15, alpha=0.6, color="steelblue")
        lims = [min(y_true.min(), y_pred.min()) - 0.01,
                max(y_true.max(), y_pred.max()) + 0.01]
        ax.plot(lims, lims, "r--", lw=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual DFT length (Å)")
        ax.set_ylabel("Predicted DFT length (Å)")
        ax.set_title(f"Parity Plot  (CV MAE = {mae:.4f} Å)")
        ax.set_aspect("equal")
        fig.tight_layout()
        savefig(fig, out_name)


def save_cv_scores_csv(scores, out_name: str = "03_cv_scores.csv") -> None:
    import pandas as pd
    df = pd.DataFrame({
        "fold": range(1, len(scores)+1),
        "mae_A": scores,
    })
    df.loc[len(df)] = ["mean", scores.mean()]
    df.loc[len(df)] = ["std",  scores.std()]
    path = FIGURES_DIR / out_name
    FIGURES_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  CV 점수 CSV 저장: {path}")


# ── 벤치마크 단계 ────────────────────────────────────────────────────

def plot_benchmark(results: dict, conditions: list[str],
                   out_name: str = "05_benchmark.png") -> None:
    """분자별 조건별 스텝 수 / CPU 시간 비교 막대 그래프."""
    molecules = list(results.keys())
    n = len(molecules)
    x = np.arange(n)
    width = 0.8 / len(conditions)
    colors = ["steelblue", "tomato", "seagreen"]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n*0.6), 7), sharex=True)

        for i, (cond, color) in enumerate(zip(conditions, colors)):
            steps = [results[m][cond].opt_steps if cond in results[m] else 0 for m in molecules]
            times = [results[m][cond].cpu_seconds if cond in results[m] else 0 for m in molecules]
            offset = (i - len(conditions)/2 + 0.5) * width
            ax1.bar(x + offset, steps, width, label=cond, color=color, alpha=0.85)
            ax2.bar(x + offset, times, width, color=color, alpha=0.85)

        ax1.set_ylabel("Optimization Steps")
        ax1.set_title("Benchmark: Optimization Steps per Molecule")
        ax1.legend()
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax2.set_ylabel("CPU Time (s)")
        ax2.set_title("Benchmark: CPU Time per Molecule")
        ax2.set_xticks(x)
        ax2.set_xticklabels(molecules, rotation=40, ha="right", fontsize=8)

        fig.tight_layout()
        savefig(fig, out_name)


def plot_acetylene_correction(data: dict, out_name: str = "06_acetylene_correction.png") -> None:
    """
    Acetylene ML 보정 결과 4-패널 시각화.

    data keys: mmff_length, ml_length, dft_length,
               mmff_steps, ml_steps, mmff_cpu, ml_cpu
               (mmff_steps=None 이면 비수렴)
    """
    mmff  = data["mmff_length"]
    ml    = data["ml_length"]
    dft   = data["dft_length"]
    delta_mmff = abs(mmff - dft)
    delta_ml   = abs(ml  - dft)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle("Acetylene: ML Bond-Length Correction vs Gaussian DFT",
                     fontsize=13, fontweight="bold", y=0.98)

        # ── Panel 1: 결합 길이 비교 (수평 막대) ─────────────────────
        ax1 = fig.add_subplot(2, 2, 1)
        labels = ["MMFF\n(RDKit)", "ML-Corrected", "DFT Reference\n(B3LYP/6-31G(d))"]
        values = [mmff, ml, dft]
        colors = ["#5B9BD5", "#ED7D31", "#70AD47"]
        bars = ax1.barh(labels, values, color=colors, edgecolor="white", height=0.5)
        ax1.set_xlim(1.18, 1.22)
        ax1.set_xlabel("C≡C Bond Length (Å)")
        ax1.set_title("Bond Length Comparison")
        for bar, val in zip(bars, values):
            ax1.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f} Å", va="center", fontsize=9)
        ax1.axvline(dft, color="#70AD47", lw=1.2, ls="--", alpha=0.6)

        # ── Panel 2: 오차 비교 (로그 스케일 막대) ────────────────────
        ax2 = fig.add_subplot(2, 2, 2)
        err_labels = ["MMFF error", "ML error"]
        err_vals   = [delta_mmff, delta_ml]
        err_colors = ["#5B9BD5", "#ED7D31"]
        bars2 = ax2.bar(err_labels, err_vals, color=err_colors, width=0.4, edgecolor="white")
        ax2.set_ylabel("|Predicted − DFT| (Å)")
        ax2.set_title("Error vs B3LYP/6-31G(d)")
        ax2.axhline(0.005, color="red", lw=1, ls="--", label="Target < 0.005 Å")
        ax2.legend(fontsize=8)
        for bar, val in zip(bars2, err_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                     f"{val:.4f} Å", ha="center", va="bottom", fontsize=9)
        reduction = (1 - delta_ml / delta_mmff) * 100
        ax2.set_title(f"Error vs DFT  ({reduction:.0f}% reduction by ML)")

        # ── Panel 3: Gaussian 최적화 스텝/시간 비교 ──────────────────
        ax3 = fig.add_subplot(2, 2, 3)
        conds  = ["MMFF\n(RDKit)", "ML-Corrected"]
        steps  = [data.get("mmff_steps"), data.get("ml_steps", 1)]
        cpus   = [data.get("mmff_cpu"), data.get("ml_cpu", 9)]
        step_display = [s if s is not None else 0 for s in steps]
        bar_colors = ["#5B9BD5", "#ED7D31"]
        b3 = ax3.bar(conds, step_display, color=bar_colors, width=0.4, edgecolor="white")
        ax3.set_ylabel("Optimization Steps")
        ax3.set_title("Gaussian Opt Steps")
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        for bar, s in zip(b3, steps):
            label = "FAIL\n(non-conv.)" if s is None else str(s)
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     label, ha="center", va="bottom", fontsize=9,
                     color="red" if s is None else "black")

        # ── Panel 4: 보정 경로 (화살표 다이어그램) ─────────────────
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_xlim(0, 3)
        ax4.set_ylim(1.195, 1.215)
        ax4.set_yticks([mmff, ml, dft])
        ax4.set_yticklabels([f"MMFF\n{mmff:.4f} Å", f"ML\n{ml:.4f} Å", f"DFT\n{dft:.4f} Å"],
                            fontsize=8)
        ax4.set_xticks([])
        ax4.set_title("Correction Pathway  (C≡C)")
        ax4.set_ylabel("Bond Length (Å)")

        # MMFF → ML 화살표
        ax4.annotate("", xy=(1.5, ml), xytext=(1.5, mmff),
                     arrowprops=dict(arrowstyle="->", color="#ED7D31", lw=2))
        ax4.text(1.6, (mmff + ml)/2, f"+{ml-mmff:.4f} Å\n(ML)", fontsize=8, color="#ED7D31")

        # ML vs DFT 잔차
        ax4.annotate("", xy=(2.3, dft), xytext=(2.3, ml),
                     arrowprops=dict(arrowstyle="<->", color="#70AD47", lw=1.5, ls="dashed"))
        ax4.text(2.35, (ml + dft)/2, f"Δ={delta_ml:.4f} Å", fontsize=8, color="#70AD47")

        # 수평 기준선
        for y, c in [(mmff, "#5B9BD5"), (ml, "#ED7D31"), (dft, "#70AD47")]:
            ax4.axhline(y, color=c, lw=0.8, ls=":")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        savefig(fig, out_name)


def save_acetylene_csv(data: dict, out_name: str = "06_acetylene_correction.csv") -> None:
    """Acetylene 보정 결과 CSV 저장."""
    import csv
    path = FIGURES_DIR / out_name
    FIGURES_DIR.mkdir(exist_ok=True)
    rows = [
        ["metric", "MMFF", "ML_corrected", "DFT_reference", "unit"],
        ["C≡C bond length",
         f"{data['mmff_length']:.6f}",
         f"{data['ml_length']:.6f}",
         f"{data['dft_length']:.6f}",
         "Angstrom"],
        ["error vs DFT",
         f"{abs(data['mmff_length'] - data['dft_length']):.6f}",
         f"{abs(data['ml_length']   - data['dft_length']):.6f}",
         "0.000000",
         "Angstrom"],
        ["error reduction (%)",
         "0",
         f"{(1 - abs(data['ml_length']-data['dft_length']) / abs(data['mmff_length']-data['dft_length']))*100:.1f}",
         "100",
         "%"],
        ["gaussian_opt_steps",
         str(data.get("mmff_steps", "FAIL")),
         str(data.get("ml_steps", "")),
         "—",
         "steps"],
        ["gaussian_cpu_time",
         str(data.get("mmff_cpu", "FAIL")),
         f"{data.get('ml_cpu', ''):.1f}" if data.get("ml_cpu") else "FAIL",
         "—",
         "seconds"],
        ["normal_termination",
         str(data.get("mmff_converged", False)),
         str(data.get("ml_converged", True)),
         "—",
         "bool"],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"  Acetylene CSV 저장: {path}")


def save_benchmark_csv(results: dict, conditions: list[str],
                        out_name: str = "05_benchmark.csv") -> None:
    import csv
    path = FIGURES_DIR / out_name
    FIGURES_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["molecule"]
        for c in conditions:
            header += [f"{c}_steps", f"{c}_cpu_s", f"{c}_ok"]
        w.writerow(header)
        for mol, cond_map in results.items():
            row = [mol]
            for c in conditions:
                if c in cond_map:
                    r = cond_map[c]
                    row += [r.opt_steps, f"{r.cpu_seconds:.1f}", r.normal_termination]
                else:
                    row += ["", "", ""]
            w.writerow(row)
    print(f"  벤치마크 CSV 저장: {path}")
