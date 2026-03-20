"""
Acetylene benchmark: MMFF vs ML-Corrected
- MMFF 파이프라인은 비수렴 여부 포함하여 실행
- ML 파이프라인 결과와 전면 비교
- figures/06_acetylene_correction.png, .csv 갱신
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from delta_chem.ml.corrector import correct_geometry
from delta_chem.chem.smiles_to_xyz import mol_to_xyz
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.chem.gaussian_runner import run_gaussian
from delta_chem.chem.log_parser import parse_log, parse_final_geometry
from delta_chem.viz import plot_acetylene_correction, save_acetylene_csv

SMILES = "C#C"
NAME   = "acetylene"
MODEL  = "models/bond_length_corrector.joblib"
OUTDIR = Path("data/raw") / NAME
OUTDIR.mkdir(parents=True, exist_ok=True)


def bond_length(mol, idx=0):
    bond = mol.GetBondWithIdx(idx)
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    conf = mol.GetConformer()
    p1, p2 = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)



def c_c_length(coords):
    """Standard orientation 결과에서 C-C 결합 길이 계산."""
    c = [(r[1], r[2], r[3]) for r in coords if r[0] == "C"]
    if len(c) < 2:
        return None
    return math.sqrt(sum((c[0][k]-c[1][k])**2 for k in range(3)))


# ── 공통: MMFF 구조 생성 ──────────────────────────────────────
print("MMFF 구조 생성...")
mol_base = Chem.MolFromSmiles(SMILES)
mol_base = Chem.AddHs(mol_base)
AllChem.EmbedMolecule(mol_base, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol_base)
d_mmff = bond_length(mol_base)
print(f"  MMFF C≡C: {d_mmff:.4f} Å")

results = {}

# ════════════════════════════════════════════
# Pipeline 1: MMFF (RDKit)
# ════════════════════════════════════════════
print("\n[1/2] MMFF 파이프라인...")
xyz1 = str(OUTDIR / f"{NAME}_rdkit.xyz")
com1 = str(OUTDIR / f"{NAME}_rdkit.com")
mol_to_xyz(mol_base, xyz1)
xyz_to_gaussian_com(xyz1, com1, title=f"{NAME} rdkit")
out1 = run_gaussian(com1)
r1   = parse_log(out1)

if r1.normal_termination:
    coords1 = parse_final_geometry(out1)
    d_dft_from_rdkit = c_c_length(coords1)
    print(f"  ✓  steps={r1.opt_steps}  cpu={r1.cpu_seconds:.1f}s")
    print(f"  DFT C≡C (from MMFF start): {d_dft_from_rdkit:.4f} Å")
    results["rdkit"] = dict(
        converged=True, steps=r1.opt_steps, cpu=r1.cpu_seconds,
        dft_length=d_dft_from_rdkit
    )
else:
    print(f"  ✗  비수렴 (steps recorded={r1.opt_steps})")
    results["rdkit"] = dict(converged=False, steps=None, cpu=None, dft_length=None)

# ════════════════════════════════════════════
# Pipeline 2: ML-Corrected
# ════════════════════════════════════════════
print("\n[2/2] ML-Corrected 파이프라인...")
mol_ml = correct_geometry(mol_base, MODEL)
d_ml   = bond_length(mol_ml)
print(f"  ML 보정 C≡C: {d_ml:.4f} Å  ({d_ml-d_mmff:+.4f} Å)")

xyz2 = str(OUTDIR / f"{NAME}_ml.xyz")
com2 = str(OUTDIR / f"{NAME}_ml.com")
mol_to_xyz(mol_ml, xyz2)
xyz_to_gaussian_com(xyz2, com2, title=f"{NAME} ml-corrected")
out2 = run_gaussian(com2)
r2   = parse_log(out2)

if r2.normal_termination:
    coords2  = parse_final_geometry(out2)
    d_dft_ml = c_c_length(coords2)
    print(f"  ✓  steps={r2.opt_steps}  cpu={r2.cpu_seconds:.1f}s")
    print(f"  DFT C≡C (from ML start):   {d_dft_ml:.4f} Å")
    results["ml"] = dict(
        converged=True, steps=r2.opt_steps, cpu=r2.cpu_seconds,
        dft_length=d_dft_ml
    )
else:
    print(f"  ✗  비수렴")
    results["ml"] = dict(converged=False, steps=None, cpu=None, dft_length=None)

# ════════════════════════════════════════════
# 요약 출력
# ════════════════════════════════════════════
print("\n" + "="*52)
print(f"{'':22s} {'MMFF':>12s}  {'ML-Corrected':>12s}")
print("="*52)

def fmt(val, fmt_str, fail="FAIL"):
    return format(val, fmt_str) if val is not None else fail

r_mmff = results["rdkit"]
r_ml   = results["ml"]

# DFT reference: 두 파이프라인 중 수렴한 값 (ML 우선)
d_dft_ref = r_ml["dft_length"] or r_mmff["dft_length"]

print(f"{'Initial C≡C (Å)':<22s} {d_mmff:>12.4f}  {d_ml:>12.4f}")
if d_dft_ref:
    err_mmff = abs(d_mmff - d_dft_ref) if d_dft_ref else None
    err_ml   = abs(d_ml   - d_dft_ref) if d_dft_ref else None
    print(f"{'Error vs DFT (Å)':<22s} {fmt(err_mmff,'>12.4f'):>12s}  {fmt(err_ml,'>12.4f'):>12s}")
print(f"{'DFT C≡C (Å)':<22s} {fmt(r_mmff['dft_length'],'>12.4f'):>12s}  {fmt(r_ml['dft_length'],'>12.4f'):>12s}")
print(f"{'Converged':<22s} {str(r_mmff['converged']):>12s}  {str(r_ml['converged']):>12s}")
print(f"{'Opt steps':<22s} {fmt(r_mmff['steps'],'>12d'):>12s}  {fmt(r_ml['steps'],'>12d'):>12s}")
print(f"{'CPU time (s)':<22s} {fmt(r_mmff['cpu'],'>12.1f'):>12s}  {fmt(r_ml['cpu'],'>12.1f'):>12s}")
if r_mmff["steps"] and r_ml["steps"]:
    step_red = (1 - r_ml["steps"] / r_mmff["steps"]) * 100
    cpu_red  = (1 - r_ml["cpu"]   / r_mmff["cpu"])   * 100
    print(f"\n  스텝 감소: {step_red:+.0f}%   CPU 감소: {cpu_red:+.0f}%")
print("="*52)

# ════════════════════════════════════════════
# 시각화 & CSV 저장
# ════════════════════════════════════════════
print("\n시각화 저장...")
viz_data = {
    "mmff_length":    d_mmff,
    "ml_length":      d_ml,
    "dft_length":     d_dft_ref,
    "mmff_steps":     r_mmff["steps"],
    "ml_steps":       r_ml["steps"],
    "mmff_cpu":       r_mmff["cpu"],
    "ml_cpu":         r_ml["cpu"],
    "mmff_converged": r_mmff["converged"],
    "ml_converged":   r_ml["converged"],
}
plot_acetylene_correction(viz_data)
save_acetylene_csv(viz_data)
print("완료.")
