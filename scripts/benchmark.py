"""
벤치마크: RDKit 단독 vs ML 보정 vs xtb 전처리 → Gaussian opt 비교

사용법:
    conda run -n delta_chem python scripts/benchmark.py
    conda run -n delta_chem python scripts/benchmark.py --conditions rdkit ml
"""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from delta_chem.chem.smiles_to_xyz import smiles_to_xyz
from delta_chem.chem.xtb_optimizer import optimize_with_xtb
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.chem.gaussian_runner import run_gaussian
from delta_chem.chem.log_parser import parse_log, GaussianResult
from delta_chem.config import XTB_EXE

MOLECULES = [
    ("methane",       "C",           0, 1),
    ("ethane",        "CC",          0, 1),
    ("propane",       "CCC",         0, 1),
    ("benzene",       "c1ccccc1",    0, 1),
    ("phenol",        "Oc1ccccc1",   0, 1),
    ("diethyl_ether", "CCOCC",       0, 1),
    ("acetone",       "CC(C)=O",     0, 1),
    ("aniline",       "Nc1ccccc1",   0, 1),
    ("naphthalene",   "c1ccc2ccccc2c1", 0, 1),
    ("cyclohexane",   "C1CCCCC1",    0, 1),
]


def _prepare_xyz(
    smiles: str, name: str, work_dir: Path, condition: str,
    charge: int, mult: int, model_path: str,
) -> str:
    import tempfile, os
    tmpdir = work_dir / "tmp"
    tmpdir.mkdir(exist_ok=True)

    rdkit_xyz = str(tmpdir / f"{name}_rdkit.xyz")
    smiles_to_xyz(smiles, rdkit_xyz, mol_name=name)

    if condition == "xtb":
        xtb_xyz = str(tmpdir / f"{name}_xtb.xyz")
        optimize_with_xtb(rdkit_xyz, xtb_xyz, charge=charge,
                          multiplicity=mult, xtb_exe=XTB_EXE)
        return xtb_xyz

    if condition == "ml":
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from delta_chem.ml.corrector import correct_geometry
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        corrected = correct_geometry(mol, model_path)
        ml_xyz = str(tmpdir / f"{name}_ml.xyz")
        conf = corrected.GetConformer()
        with open(ml_xyz, "w") as f:
            f.write(f"{corrected.GetNumAtoms()}\n{name}\n")
            for i in range(corrected.GetNumAtoms()):
                a = corrected.GetAtomWithIdx(i)
                p = conf.GetAtomPosition(i)
                f.write(f"{a.GetSymbol():2s}  {p.x:12.6f}  {p.y:12.6f}  {p.z:12.6f}\n")
        return ml_xyz

    return rdkit_xyz  # condition == "rdkit"


def run_benchmark(
    conditions: list[str],
    route: str,
    nproc: int,
    mem: str,
    output_dir: str,
    model_path: str,
) -> None:
    out_dir = Path(output_dir)
    all_results: dict[str, dict[str, GaussianResult]] = {}

    for name, smiles, charge, mult in MOLECULES:
        print(f"\n[{name}]")
        work_dir = out_dir / name
        work_dir.mkdir(parents=True, exist_ok=True)
        all_results[name] = {}

        for cond in conditions:
            com = str(work_dir / f"{name}_{cond}.com")
            try:
                input_xyz = _prepare_xyz(smiles, name, work_dir, cond,
                                         charge, mult, model_path)
                xyz_to_gaussian_com(input_xyz, com, route=route,
                                    title=f"{name}[{cond}]",
                                    charge=charge, multiplicity=mult,
                                    nproc=nproc, mem=mem)
                print(f"  [{cond}] Gaussian 실행 중...", flush=True)
                out_path = run_gaussian(com)
                res = parse_log(out_path)
                all_results[name][cond] = res
                print(f"  [{cond}] steps={res.opt_steps}  "
                      f"cpu={res.cpu_seconds:.0f}s  ok={res.normal_termination}")
            except Exception as e:
                print(f"  [{cond}] 실패: {e}")

    _print_table(all_results, conditions)
    _save_csv(all_results, conditions, str(out_dir / "summary.csv"))

    from delta_chem.viz import plot_benchmark, save_benchmark_csv
    plot_benchmark(all_results, conditions)
    save_benchmark_csv(all_results, conditions)


def _print_table(results, conditions):
    print(f"\n{'='*70}")
    print("벤치마크 결과")
    print(f"{'='*70}")
    header = f"{'분자':<16}" + "".join(f"  {c}(스텝/초)" for c in conditions)
    print(header)
    print("-" * 70)
    for name, cond_map in results.items():
        row = f"{name:<16}"
        for c in conditions:
            if c in cond_map:
                r = cond_map[c]
                ok = "✓" if r.normal_termination else "✗"
                row += f"  {r.opt_steps:>3}스텝/{r.cpu_seconds:>5.0f}s{ok}"
            else:
                row += "  N/A"
        print(row)
    print(f"{'='*70}\n")


def _save_csv(results, conditions, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["molecule"] +
                   [f"{c}_steps" for c in conditions] +
                   [f"{c}_cpu_s" for c in conditions] +
                   [f"{c}_ok" for c in conditions])
        for name, cond_map in results.items():
            w.writerow([name] +
                       [cond_map[c].opt_steps if c in cond_map else "" for c in conditions] +
                       [cond_map[c].cpu_seconds if c in cond_map else "" for c in conditions] +
                       [cond_map[c].normal_termination if c in cond_map else "" for c in conditions])
    print(f"CSV 저장: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["rdkit", "ml"],
                        choices=["rdkit", "xtb", "ml"])
    parser.add_argument("--route", default="#p opt B3LYP/6-31G(d)")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--mem", default="4GB")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--model-path", default="models/bond_length_corrector.joblib")
    args = parser.parse_args()
    run_benchmark(args.conditions, args.route, args.nproc,
                  args.mem, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()
