"""전체 파이프라인 스모크 테스트 (xtb 포함)."""
import tempfile
from pathlib import Path
from delta_chem.chem.smiles_to_xyz import smiles_to_xyz
from delta_chem.chem.xtb_optimizer import optimize_with_xtb
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.config import XTB_EXE

MOLECULES = [
    ("CCO",       "ethanol",     0, 1),
    ("c1ccccc1",  "benzene",     0, 1),
    ("CC(=O)O",   "acetic_acid", 0, 1),
]

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    for smiles, name, charge, mult in MOLECULES:
        print(f"\n{'='*40}\n{name}  ({smiles})")
        rdkit_xyz = str(tmpdir / f"{name}_rdkit.xyz")
        xtb_xyz   = str(tmpdir / f"{name}_xtb.xyz")
        com_path  = f"{name}.com"

        smiles_to_xyz(smiles, rdkit_xyz, mol_name=name)
        print(f"  [1] RDKit OK")

        optimize_with_xtb(rdkit_xyz, xtb_xyz, charge=charge,
                          multiplicity=mult, xtb_exe=XTB_EXE)
        print(f"  [2] xtb OK")

        xyz_to_gaussian_com(xtb_xyz, com_path, charge=charge,
                            multiplicity=mult, title=name)
        print(f"  [3] .com OK → {com_path}")

print("\nAll tests passed.")
