"""
훈련 미사용 분자 벤치마크: MMFF vs GBM vs GNN

4가지 초기 구조를 각각 Gaussian B3LYP/6-31G(d)로 최적화하여
최적화 스텝 수 / CPU 시간을 비교한다.

사용법:
    conda run -n delta_chem python scripts/benchmark_new_mols.py
    conda run -n delta_chem python scripts/benchmark_new_mols.py --nproc 20 --mem 56GB
    conda run -n delta_chem python scripts/benchmark_new_mols.py --rerun
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from delta_chem.ml.corrector import correct_geometry
from scripts.gnn.corrector_gnn import correct_geometry_gnn
from delta_chem.chem.smiles_to_xyz import mol_to_xyz
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.chem.gaussian_runner import run_gaussian
from delta_chem.chem.log_parser import parse_log, parse_final_geometry

# ── 모델 경로 ──────────────────────────────────────────────────────────────────
MODEL_OLD  = "models/bond_length_corrector_delta.joblib"           # GBM 50개 분자
MODEL_EXP  = "models/bond_length_corrector_delta_expanded.joblib"  # GBM 120개 분자
MODEL_GNN  = "models/bond_length_corrector_gnn.joblib"             # GNN 120개 분자
MODEL_FULL = "models/bond_length_corrector_delta_full.joblib"      # GBM 659개 분자
MODEL_GNN_FULL = "models/bond_length_corrector_gnn_full.joblib"    # GNN 659개 분자

# ── 벤치마크 분자 (훈련셋 미포함 확인 완료) ────────────────────────────────────
MOLECULES = [
    # (name, SMILES, 설명)
    ("cyclopentane",      "C1CCCC1",         "5원 고리 알케인 (훈련: cyclohexane만)"),
    ("1-butanol",         "CCCCO",           "C4 알코올 (훈련: 최대 1-propanol)"),
    ("acetophenone",      "CC(=O)c1ccccc1",  "방향족 케톤 Ar-C=O (훈련 미포함)"),
    ("acrolein",          "C=CC=O",          "공액 카보닐 C=C-C=O (훈련 미포함)"),
    ("dimethylsulfoxide", "CS(C)=O",         "설폭사이드 S=O (훈련 미포함)"),
]

PIPELINES = [
    ("rdkit",     None,          None),  # MMFF baseline
    ("gbm_50",    MODEL_OLD,     "gbm"), # GBM 50개 분자
    ("gbm_full",  MODEL_FULL,    "gbm"), # GBM 659개 분자
    ("gnn_full",  MODEL_GNN_FULL,"gnn"), # GNN 659개 분자
]

OUTDIR = Path("data/raw/benchmark_new")
FIGDIR = Path("figures")


def mmff_mol(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def run_gaussian_job(name, smiles, tag, mol_obj, work_dir, nproc, mem):
    """XYZ → .com → Gaussian 실행. (name, tag) 기준으로 이미 완료된 경우 건너뜀."""
    work_dir.mkdir(parents=True, exist_ok=True)

    # 완료 여부 확인
    for suffix in (".log", ".out"):
        existing = work_dir / f"{name}_{tag}{suffix}"
        if existing.exists() and existing.stat().st_size > 0:
            text = existing.read_text(encoding="utf-8", errors="replace")
            if "Normal termination" in text:
                res = parse_log(str(existing))
                coords = parse_final_geometry(str(existing))
                print(f"    이미 완료, 건너뜀 (steps={res.opt_steps})")
                return res, coords

    xyz_p = str(work_dir / f"{name}_{tag}.xyz")
    com_p = str(work_dir / f"{name}_{tag}.com")
    mol_to_xyz(mol_obj, xyz_p, f"{name} {tag}")
    xyz_to_gaussian_com(xyz_p, com_p, title=f"{name} {tag}",
                        nproc=nproc, mem=mem)
    out = run_gaussian(com_p)
    res = parse_log(out)
    coords = parse_final_geometry(out) if res.normal_termination else []
    return res, coords


def run_benchmark(nproc: int, mem: str, rerun: bool) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(exist_ok=True)

    # --rerun: 기존 결과 삭제
    if rerun and OUTDIR.exists():
        import shutil
        shutil.rmtree(OUTDIR)
        OUTDIR.mkdir(parents=True)
        print("기존 결과 삭제 후 재실행\n")

    records = []

    for mol_name, smiles, note in MOLECULES:
        print(f"\n{'='*60}")
        print(f"  {mol_name}  |  {smiles}")
        print(f"  {note}")
        print(f"{'='*60}")

        work_dir = OUTDIR / mol_name

        # MMFF 기본 구조 생성
        mol_base = mmff_mol(smiles)
        if mol_base is None:
            print("  ✗ MMFF 좌표 생성 실패, 건너뜀")
            continue

        for tag, model_path, model_type in PIPELINES:
            print(f"\n  [{tag.upper()}] ", end="", flush=True)

            # 보정 구조 생성
            if model_path is not None:
                if not Path(model_path).exists():
                    print(f"모델 없음 ({model_path}), 건너뜀")
                    continue
                if model_type == "gnn":
                    mol_obj = correct_geometry_gnn(mol_base, model_path)
                else:
                    mol_obj = correct_geometry(mol_base, model_path)
                print("보정 적용 완료. Gaussian 실행 중...", flush=True)
            else:
                mol_obj = mol_base
                print("Gaussian 실행 중...", flush=True)

            try:
                res, coords = run_gaussian_job(
                    mol_name, smiles, tag, mol_obj, work_dir, nproc, mem
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                records.append({
                    "molecule": mol_name, "smiles": smiles, "note": note,
                    "pipeline": tag, "converged": False,
                    "opt_steps": None, "cpu_s": None, "elapsed_s": None,
                })
                continue

            status = "✓" if res.normal_termination else "✗ 비수렴"
            print(f"    {status}  steps={res.opt_steps}  cpu={res.cpu_seconds:.1f}s")
            records.append({
                "molecule":  mol_name,
                "smiles":    smiles,
                "note":      note,
                "pipeline":  tag,
                "converged": res.normal_termination,
                "opt_steps": res.opt_steps if res.normal_termination else None,
                "cpu_s":     res.cpu_seconds if res.normal_termination else None,
            })

    return pd.DataFrame(records)


def report_and_plot(df: pd.DataFrame) -> None:
    csv_path = FIGDIR / "09_benchmark_new_mols.csv"
    df.to_csv(csv_path, index=False)

    print("\n\n" + "=" * 60)
    print("  벤치마크 결과 요약")
    print("=" * 60)

    pivot = df.pivot_table(
        index="molecule", columns="pipeline",
        values=["converged", "opt_steps", "cpu_s"],
        aggfunc="first",
    )
    print(pivot.to_string())

    # 개선율 (rdkit 기준)
    df_ref = df[df["pipeline"] == "rdkit"].set_index("molecule")
    cmp_tags = [t for t, _, _ in PIPELINES if t != "rdkit"]
    print("\n  스텝 개선율 (rdkit 기준):")
    header = f"  {'분자':<22s} {'rdkit':>6}" + "".join(f" {t:>10}" for t in cmp_tags)
    print(header)
    print("  " + "-" * (30 + 11 * len(cmp_tags)))

    for mol_name, _, _ in MOLECULES:
        if mol_name not in df_ref.index:
            continue
        r = df_ref.loc[mol_name]
        if not r["converged"] or r["opt_steps"] is None:
            continue
        row = f"  {mol_name:<22s} {int(r['opt_steps']):>6}"
        for tag in cmp_tags:
            sub = df[(df["pipeline"] == tag) & (df["molecule"] == mol_name)]
            if sub.empty or not sub.iloc[0]["converged"]:
                row += f"  {'✗':>8}"
            else:
                steps = int(sub.iloc[0]["opt_steps"])
                imp = (1 - steps / r["opt_steps"]) * 100
                row += f"  {steps:>4} ({imp:+.0f}%)"
        print(row)

    print(f"\n  CSV: {csv_path}")

    # ── 시각화 ──────────────────────────────────────────────────────────────
    pipeline_labels = {
        "rdkit":    "MMFF (RDKit)",
        "gbm_50":   "GBM-50",
        "gbm_full": "GBM-659",
        "gnn_full": "GNN-659",
    }
    colors = {"rdkit": "#5B9BD5", "gbm_50": "#ED7D31", "gbm_full": "#70AD47", "gnn_full": "#C00000"}

    conv_mols = [m for m, _, _ in MOLECULES
                 if df[(df["molecule"] == m) & (df["pipeline"] == "rdkit") & df["converged"]].shape[0] > 0]

    if not conv_mols:
        print("수렴 분자 없음 — 그래프 생략")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Benchmark: Untrained Molecules — MMFF vs GBM vs GNN",
                 fontsize=12, fontweight="bold")

    tags = [t for t, _, _ in PIPELINES]
    x = np.arange(len(conv_mols))
    n = len(tags)
    w = 0.18

    for ax_idx, metric in enumerate(["opt_steps", "cpu_s"]):
        ax = axes[ax_idx]
        for k, tag in enumerate(tags):
            vals = []
            for mol in conv_mols:
                sub = df[(df["molecule"] == mol) & (df["pipeline"] == tag)]
                v = sub.iloc[0][metric] if not sub.empty and sub.iloc[0]["converged"] else 0
                vals.append(v or 0)
            offset = (k - (n - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w,
                          label=pipeline_labels[tag],
                          color=colors[tag], edgecolor="white")
            for bar, v in zip(bars, vals):
                if v > 0:
                    label = str(int(v)) if metric == "opt_steps" else f"{v:.0f}s"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.05 if metric == "opt_steps" else 1),
                            label, ha="center", va="bottom", fontsize=7,
                            color=colors[tag])

        ax.set_xticks(x)
        ax.set_xticklabels(conv_mols, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Optimization Steps" if metric == "opt_steps" else "CPU Time (s)")
        ax.set_title("최적화 스텝" if metric == "opt_steps" else "CPU 시간")
        if metric == "opt_steps":
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)

    # 합계 개선율 텍스트
    ref_sum = df[(df["pipeline"] == "rdkit") & (df["molecule"].isin(conv_mols)) & df["converged"]]["opt_steps"].sum()
    summary_parts = []
    for tag in cmp_tags:
        tag_sum = df[(df["pipeline"] == tag) & (df["molecule"].isin(conv_mols)) & df["converged"]]["opt_steps"].sum()
        if ref_sum and tag_sum > 0:
            imp = (1 - tag_sum / ref_sum) * 100
            summary_parts.append(f"{pipeline_labels[tag]}: {imp:+.1f}%")
    fig.text(0.5, 0.01, "  |  ".join(summary_parts),
             ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    png_path = FIGDIR / "09_benchmark_new_mols.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  그래프: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="훈련 미사용 분자 3-way 벤치마크")
    parser.add_argument("--nproc", type=int, default=20)
    parser.add_argument("--mem",   default="56GB")
    parser.add_argument("--rerun", action="store_true",
                        help="기존 Gaussian 결과 삭제 후 전체 재실행")
    args = parser.parse_args()

    print(f"nproc={args.nproc}, mem={args.mem}")
    print(f"비교: rdkit / ml-50({MODEL_OLD}) / ml-120({MODEL_EXP})")

    df = run_benchmark(args.nproc, args.mem, args.rerun)
    report_and_plot(df)
    print("\n완료.")


if __name__ == "__main__":
    main()
