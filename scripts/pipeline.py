"""
delta_chem pipeline: SMILES → (RDKit+MMFF) → [ML 보정] → Gaussian .com

사용법:
    conda run -n delta_chem python scripts/pipeline.py "CCO" --name ethanol
    conda run -n delta_chem python scripts/pipeline.py "CCO" --name ethanol --ml-correct
"""
import argparse
import tempfile
import os
from pathlib import Path

from delta_chem.chem.smiles_to_xyz import smiles_to_xyz
from delta_chem.chem.xtb_optimizer import optimize_with_xtb
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.config import XTB_EXE


def run_pipeline(
    smiles: str,
    name: str,
    output_dir: str = ".",
    charge: int = 0,
    multiplicity: int = 1,
    route: str = "#p opt freq B3LYP/6-31G(d)",
    nproc: int = 4,
    mem: str = "4GB",
    use_xtb: bool = False,
    ml_correct: bool = False,
    model_path: str = "models/bond_length_corrector.joblib",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        rdkit_xyz = os.path.join(tmpdir, f"{name}_rdkit.xyz")
        xtb_xyz   = os.path.join(tmpdir, f"{name}_xtb.xyz")
        ml_xyz    = os.path.join(tmpdir, f"{name}_ml.xyz")

        # Step 1: SMILES → RDKit 3D
        print(f"[1] RDKit ETKDG+MMFF ...", flush=True)
        smiles_to_xyz(smiles, rdkit_xyz, mol_name=name)
        input_xyz = rdkit_xyz

        # Step 2 (선택): ML bond length 보정
        if ml_correct and Path(model_path).exists():
            print(f"[2] ML bond length correction ({model_path}) ...", flush=True)
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from delta_chem.ml.corrector import correct_geometry
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            corrected = correct_geometry(mol, model_path)
            conf = corrected.GetConformer()
            atoms = [corrected.GetAtomWithIdx(i) for i in range(corrected.GetNumAtoms())]
            with open(ml_xyz, "w") as f:
                f.write(f"{corrected.GetNumAtoms()}\n{name}\n")
                for atom in atoms:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    f.write(f"{atom.GetSymbol():2s}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}\n")
            input_xyz = ml_xyz
        elif use_xtb:
            print(f"[2] xtb pre-optimization ...", flush=True)
            optimize_with_xtb(rdkit_xyz, xtb_xyz, charge=charge,
                              multiplicity=multiplicity, xtb_exe=XTB_EXE)
            input_xyz = xtb_xyz
        else:
            print(f"[2] 추가 전처리 없음 (RDKit 구조 사용)")

        # Step 3: Gaussian .com 생성
        com_path = out_dir / f"{name}.com"
        print(f"[3] Gaussian .com 생성 → {com_path}", flush=True)
        xyz_to_gaussian_com(
            input_xyz, str(com_path),
            route=route, title=f"{name} | {smiles}",
            charge=charge, multiplicity=multiplicity,
            nproc=nproc, mem=mem,
        )

    print(f"\n완료: {com_path}")
    return com_path


def main():
    parser = argparse.ArgumentParser(description="delta_chem pipeline")
    parser.add_argument("smiles")
    parser.add_argument("--name", default="molecule")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--route", default="#p opt freq B3LYP/6-31G(d)")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--mem", default="4GB")
    parser.add_argument("--use-xtb", action="store_true")
    parser.add_argument("--ml-correct", action="store_true",
                        help="ML 보정 모델 적용 (models/ 에 모델 필요)")
    parser.add_argument("--model-path", default="models/bond_length_corrector.joblib")
    args = parser.parse_args()
    run_pipeline(
        smiles=args.smiles, name=args.name, output_dir=args.output_dir,
        charge=args.charge, multiplicity=args.mult, route=args.route,
        nproc=args.nproc, mem=args.mem,
        use_xtb=args.use_xtb, ml_correct=args.ml_correct, model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
