"""Debugs parse_final_geometry and extract_bond_features."""
import sys
sys.path.insert(0, "src")

from delta_chem.chem.log_parser import parse_final_geometry
from delta_chem.ml.feature_extractor import extract_bond_features
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. 파서 테스트
print("=== parse_final_geometry (benzene) ===")
atoms = parse_final_geometry("data/raw/benzene/benzene_rdkit.out")
print(f"  원자 수: {len(atoms)}")
for a in atoms[:4]:
    print(f"  {a}")

# 2. RDKit 원자 수
print("\n=== RDKit benzene atom count ===")
mol = Chem.MolFromSmiles("c1ccccc1")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
print(f"  RDKit atoms: {mol.GetNumAtoms()}")

# 3. feature 추출
print("\n=== extract_bond_features (benzene) ===")
df = extract_bond_features("c1ccccc1", "data/raw/benzene/benzene_rdkit.out", "benzene")
print(df if not df.empty else "  EMPTY - 실패")
