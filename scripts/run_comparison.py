"""에탄올로 rdkit vs ml 조건 직접 비교."""
from pathlib import Path
from delta_chem.chem.smiles_to_xyz import smiles_to_xyz
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.chem.gaussian_runner import run_gaussian
from delta_chem.chem.log_parser import parse_log
from delta_chem.ml.corrector import correct_geometry
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile, os

SMILES    = "CCO"
NAME      = "ethanol"
ROUTE     = "#p opt B3LYP/6-31G(d)"
NPROC     = 4
MEM       = "4GB"
MODEL     = "models/bond_length_corrector.joblib"
OUT_DIR   = Path("data/raw/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

results = {}

with tempfile.TemporaryDirectory() as tmp:
    for cond in ["rdkit", "ml"]:
        rdkit_xyz = os.path.join(tmp, f"{NAME}_rdkit.xyz")
        smiles_to_xyz(SMILES, rdkit_xyz, mol_name=NAME)

        if cond == "ml":
            mol = Chem.MolFromSmiles(SMILES)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            corrected = correct_geometry(mol, MODEL)
            conf = corrected.GetConformer()
            ml_xyz = os.path.join(tmp, f"{NAME}_ml.xyz")
            with open(ml_xyz, "w") as f:
                f.write(f"{corrected.GetNumAtoms()}\n{NAME}\n")
                for i in range(corrected.GetNumAtoms()):
                    a = corrected.GetAtomWithIdx(i)
                    p = conf.GetAtomPosition(i)
                    f.write(f"{a.GetSymbol():2s}  {p.x:12.6f}  {p.y:12.6f}  {p.z:12.6f}\n")
            input_xyz = ml_xyz
        else:
            input_xyz = rdkit_xyz

        com = str(OUT_DIR / f"{NAME}_{cond}.com")
        xyz_to_gaussian_com(input_xyz, com, route=ROUTE, title=f"{NAME}[{cond}]",
                            charge=0, multiplicity=1, nproc=NPROC, mem=MEM)

        print(f"[{cond}] Gaussian 실행 중...", flush=True)
        out_path = run_gaussian(com)
        res = parse_log(out_path)
        results[cond] = res
        print(f"[{cond}] steps={res.opt_steps}  cpu={res.cpu_seconds:.1f}s  ok={res.normal_termination}")

print("\n" + "="*50)
print(f"{'조건':<10} {'스텝':>6} {'CPU(s)':>8} {'정상종료':>8}")
print("-"*50)
for cond, r in results.items():
    print(f"{cond:<10} {r.opt_steps:>6} {r.cpu_seconds:>8.1f} {str(r.normal_termination):>8}")
if "rdkit" in results and "ml" in results:
    dr = results["rdkit"]
    dm = results["ml"]
    step_diff = dr.opt_steps - dm.opt_steps
    time_diff = dr.cpu_seconds - dm.cpu_seconds
    print("-"*50)
    print(f"{'ML 단축':<10} {step_diff:>+6} {time_diff:>+8.1f}")
print("="*50)

# ── 그래프 & CSV ──────────────────────────────────────────────────────
from delta_chem.chem.log_parser import GaussianResult
from delta_chem.viz import plot_benchmark, save_benchmark_csv
nested = {NAME: results}
plot_benchmark(nested, list(results.keys()), out_name="05_benchmark_ethanol.png")
save_benchmark_csv(nested, list(results.keys()), out_name="05_benchmark_ethanol.csv")
