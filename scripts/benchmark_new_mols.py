"""
훈련 미사용 5개 분자 벤치마크: MMFF vs ML-Corrected

선정 기준:
  - 훈련 세트에 없는 분자
  - 다양한 작용기/결합 유형 포함
  - Gaussian 09 계산 가능한 소분자
"""
import sys, os, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from delta_chem.ml.corrector import correct_geometry
from delta_chem.chem.smiles_to_xyz import mol_to_xyz
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.chem.gaussian_runner import run_gaussian
from delta_chem.chem.log_parser import parse_log, parse_final_geometry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MODEL   = "models/bond_length_corrector.joblib"
OUTDIR  = Path("data/raw/benchmark_new")
FIGDIR  = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

# ── 벤치마크 대상 분자 ────────────────────────────────────────────────
# 훈련 세트에 없는 다양한 작용기
MOLECULES = [
    ("cyclopentane",  "C1CCCC1",          "5원 고리 알케인 (훈련: cyclohexane만 있음)"),
    ("1-butanol",     "CCCCO",            "탄소 4개 알코올 (훈련: 최대 1-propanol)"),
    ("acetophenone",  "CC(=O)c1ccccc1",   "방향족 케톤 (Ar-C=O, 훈련에 없음)"),
    ("acrolein",      "C=CC=O",           "공액 카보닐 (C=C-C=O, 훈련에 없음)"),
    ("dimethylsulfoxide", "CS(C)=O",      "설폭사이드 (S=O 결합, 훈련에 없음)"),
]



def run_pipeline(name, smiles, tag, mol_obj, work_dir):
    """단일 파이프라인 실행 → GaussianResult 반환"""
    work_dir.mkdir(parents=True, exist_ok=True)
    xyz_p = str(work_dir / f"{name}_{tag}.xyz")
    com_p = str(work_dir / f"{name}_{tag}.com")
    mol_to_xyz(mol_obj, xyz_p, f"{name} {tag}")
    xyz_to_gaussian_com(xyz_p, com_p, title=f"{name} {tag}")
    out   = run_gaussian(com_p)
    res   = parse_log(out)
    coords = parse_final_geometry(out) if res.normal_termination else []
    return res, coords


# ════════════════════════════════════════════════════════════════
# 메인 루프
# ════════════════════════════════════════════════════════════════
records = []   # CSV 저장용

for mol_name, smiles, note in MOLECULES:
    print(f"\n{'='*58}")
    print(f"  {mol_name}  ({smiles})")
    print(f"  {note}")
    print(f"{'='*58}")

    work_dir = OUTDIR / mol_name

    # 1. MMFF 구조
    mol_base = Chem.MolFromSmiles(smiles)
    mol_base = Chem.AddHs(mol_base)
    if AllChem.EmbedMolecule(mol_base, AllChem.ETKDGv3()) != 0:
        print("  ✗ ETKDGv3 좌표 생성 실패, 건너뜀")
        continue
    AllChem.MMFFOptimizeMolecule(mol_base)

    # 2. ML 보정 구조
    mol_ml = correct_geometry(mol_base, MODEL)

    # 3. 두 파이프라인 Gaussian 실행
    for tag, mol_obj in [("rdkit", mol_base), ("ml", mol_ml)]:
        print(f"\n  [{tag.upper()}] Gaussian 실행 중...", flush=True)
        try:
            res, coords = run_pipeline(mol_name, smiles, tag, mol_obj, work_dir)
        except Exception as e:
            print(f"    ERROR: {e}")
            records.append({
                "molecule": mol_name, "smiles": smiles, "note": note,
                "pipeline": tag, "converged": False,
                "opt_steps": None, "cpu_s": None,
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


# ════════════════════════════════════════════════════════════════
# 결과 정리
# ════════════════════════════════════════════════════════════════
df = pd.DataFrame(records)
print("\n\n" + "="*58)
print("  벤치마크 결과 요약")
print("="*58)

pivot = df.pivot(index="molecule", columns="pipeline", values=["converged","opt_steps","cpu_s"])
print(pivot.to_string())

# 개선율 계산
df_rdkit = df[df["pipeline"] == "rdkit"].set_index("molecule")
df_ml    = df[df["pipeline"] == "ml"].set_index("molecule")

print("\n  스텝/시간 개선율 (ML vs MMFF):")
for mol in df_rdkit.index:
    r, m = df_rdkit.loc[mol], df_ml.loc[mol]
    if r["converged"] and m["converged"] and r["opt_steps"] and m["opt_steps"]:
        step_d = r["opt_steps"] - m["opt_steps"]
        step_r = (1 - m["opt_steps"] / r["opt_steps"]) * 100
        cpu_r  = (1 - m["cpu_s"] / r["cpu_s"]) * 100
        print(f"    {mol:<20s}  steps: {r['opt_steps']}→{m['opt_steps']} ({step_r:+.0f}%)  "
              f"cpu: {r['cpu_s']:.0f}s→{m['cpu_s']:.0f}s ({cpu_r:+.0f}%)")
    else:
        rc = "✓" if r["converged"] else "✗"
        mc = "✓" if m["converged"] else "✗"
        print(f"    {mol:<20s}  rdkit={rc}  ml={mc}")


# ════════════════════════════════════════════════════════════════
# CSV 저장
# ════════════════════════════════════════════════════════════════
csv_path = FIGDIR / "07_benchmark_new_mols.csv"
df.to_csv(csv_path, index=False)
print(f"\n  CSV 저장: {csv_path}")


# ════════════════════════════════════════════════════════════════
# 시각화
# ════════════════════════════════════════════════════════════════
conv_mols = [m for m in df_rdkit.index
             if df_rdkit.loc[m,"converged"] and df_ml.loc[m,"converged"]]

if conv_mols:
    rdkit_steps = [df_rdkit.loc[m,"opt_steps"] for m in conv_mols]
    ml_steps    = [df_ml.loc[m,"opt_steps"]    for m in conv_mols]
    rdkit_cpu   = [df_rdkit.loc[m,"cpu_s"]     for m in conv_mols]
    ml_cpu      = [df_ml.loc[m,"cpu_s"]        for m in conv_mols]

    x     = np.arange(len(conv_mols))
    w     = 0.35
    colors = {"rdkit": "#5B9BD5", "ml": "#ED7D31"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Benchmark: Untrained Molecules — MMFF vs ML-Corrected",
                 fontsize=13, fontweight="bold")

    # ── 왼쪽: 최적화 스텝 ─────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x - w/2, rdkit_steps, w, label="MMFF (RDKit)",  color=colors["rdkit"], edgecolor="white")
    b2 = ax.bar(x + w/2, ml_steps,    w, label="ML-Corrected",  color=colors["ml"],    edgecolor="white")
    ax.set_ylabel("Optimization Steps")
    ax.set_title("Gaussian Optimization Steps")
    ax.set_xticks(x)
    ax.set_xticklabels(conv_mols, rotation=20, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8,
                color=colors["rdkit"])
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8,
                color=colors["ml"])

    # ── 오른쪽: CPU 시간 ──────────────────────────────────────────
    ax2 = axes[1]
    b3 = ax2.bar(x - w/2, rdkit_cpu, w, label="MMFF (RDKit)", color=colors["rdkit"], edgecolor="white")
    b4 = ax2.bar(x + w/2, ml_cpu,    w, label="ML-Corrected", color=colors["ml"],    edgecolor="white")
    ax2.set_ylabel("CPU Time (s)")
    ax2.set_title("Gaussian CPU Time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conv_mols, rotation=20, ha="right", fontsize=9)
    ax2.legend()
    for bar in b3:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{bar.get_height():.0f}s", ha="center", va="bottom", fontsize=8,
                 color=colors["rdkit"])
    for bar in b4:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{bar.get_height():.0f}s", ha="center", va="bottom", fontsize=8,
                 color=colors["ml"])

    # 개선율 텍스트 박스
    total_r = sum(rdkit_steps); total_m = sum(ml_steps)
    total_cr = sum(rdkit_cpu);  total_cm = sum(ml_cpu)
    step_imp = (1 - total_m/total_r)*100 if total_r else 0
    cpu_imp  = (1 - total_cm/total_cr)*100 if total_cr else 0
    fig.text(0.5, 0.01,
             f"총 {len(conv_mols)}개 분자 합계 |  Steps: {total_r}→{total_m} ({step_imp:+.1f}%)  "
             f"CPU: {total_cr:.0f}s→{total_cm:.0f}s ({cpu_imp:+.1f}%)",
             ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    png_path = FIGDIR / "07_benchmark_new_mols.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  그래프 저장: {png_path}")

print("\n완료.")
