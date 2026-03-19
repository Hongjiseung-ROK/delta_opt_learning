"""
MMFF 좌표 + Gaussian DFT 최종 좌표 → bond feature DataFrame

각 결합마다 한 행을 생성한다:
  elem1, elem2, bond_order, hybridization_1, hybridization_2,
  is_in_ring, ring_size, mmff_length → dft_length  (학습 타겟)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from delta_chem.chem.smiles_to_xyz import smiles_to_xyz
from delta_chem.chem.log_parser import parse_final_geometry


_HYB_MAP = {
    Chem.rdchem.HybridizationType.SP:  "SP",
    Chem.rdchem.HybridizationType.SP2: "SP2",
    Chem.rdchem.HybridizationType.SP3: "SP3",
}


def extract_bond_features(
    smiles: str,
    gaussian_out_path: str,
    mol_name: str = "",
) -> pd.DataFrame:
    """
    한 분자에 대해 bond feature 행들을 반환한다.

    원자 인덱스 정렬 보장:
    - smiles_to_xyz()가 H 추가 후 mol의 원자 순서대로 XYZ를 씀
    - gaussian_writer.py가 동일 XYZ 순서로 .com을 생성
    - Gaussian의 Standard orientation Center Number = 해당 순서 그대로
    따라서 RDKit mol의 GetAtomWithIdx(i) ↔ Gaussian Center i+1 이 대응된다.
    """
    # 1. RDKit mol 생성 + MMFF 최적화
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.DataFrame()
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return pd.DataFrame()
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()

    mmff_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    # 2. DFT 최종 좌표
    dft_atoms = parse_final_geometry(gaussian_out_path)
    if not dft_atoms:
        return pd.DataFrame()
    if len(dft_atoms) != mol.GetNumAtoms():
        return pd.DataFrame()  # 원자 수 불일치: 건너뜀

    dft_coords = np.array([[x, y, z] for _, x, y, z in dft_atoms])

    # 3. 링 정보
    ring_info = mol.GetRingInfo()

    rows = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)

        elem_pair = sorted([atom_i.GetSymbol(), atom_j.GetSymbol()])
        bond_order = _bond_order(bond.GetBondType())
        hyb_i = _HYB_MAP.get(atom_i.GetHybridization(), "OTHER")
        hyb_j = _HYB_MAP.get(atom_j.GetHybridization(), "OTHER")

        # 링 소속 여부 및 최소 링 크기
        in_ring = bond.IsInRing()
        ring_size = 0
        if in_ring:
            sizes = [r for r in ring_info.BondRingSizes(bond.GetIdx())]
            ring_size = min(sizes) if sizes else 0

        mmff_len = float(np.linalg.norm(mmff_coords[i] - mmff_coords[j]))
        dft_len  = float(np.linalg.norm(dft_coords[i]  - dft_coords[j]))

        rows.append({
            "mol_name":        mol_name,
            "smiles":          smiles,
            "bond_idx":        bond.GetIdx(),
            "elem1":           elem_pair[0],
            "elem2":           elem_pair[1],
            "bond_order":      bond_order,
            "hybridization_1": hyb_i if atom_i.GetSymbol() == elem_pair[0] else hyb_j,
            "hybridization_2": hyb_j if atom_i.GetSymbol() == elem_pair[0] else hyb_i,
            "is_in_ring":      int(in_ring),
            "ring_size":       ring_size,
            "mmff_length":     mmff_len,
            "dft_length":      dft_len,
        })

    return pd.DataFrame(rows)


def build_dataset(
    results_dir: str,
    molecule_list: list[tuple[str, str]],  # [(name, smiles), ...]
    condition: str = "rdkit",              # "rdkit" or "xtb"
    output_csv: str = "data/features/bond_features.csv",
) -> pd.DataFrame:
    """
    results_dir/<name>/<name>_<condition>.out 파일들을 순회하며
    bond feature CSV를 만든다.
    """
    results_dir = Path(results_dir)
    all_rows = []

    for name, smiles in molecule_list:
        out_path = results_dir / name / f"{name}_{condition}.out"
        if not out_path.exists():
            print(f"  [skip] {name}: {out_path} 없음")
            continue

        df = extract_bond_features(smiles, str(out_path), mol_name=name)
        if df.empty:
            print(f"  [skip] {name}: feature 추출 실패")
            continue

        all_rows.append(df)
        print(f"  [ok]   {name}: {len(df)}개 결합")

    if not all_rows:
        print("추출된 데이터 없음.")
        return pd.DataFrame()

    full_df = pd.concat(all_rows, ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_csv, index=False)
    print(f"\n저장: {output_csv}  ({len(full_df)}행)")
    return full_df


def _bond_order(bond_type) -> float:
    from rdkit.Chem import rdchem
    return {
        rdchem.BondType.SINGLE:    1.0,
        rdchem.BondType.DOUBLE:    2.0,
        rdchem.BondType.TRIPLE:    3.0,
        rdchem.BondType.AROMATIC:  1.5,
    }.get(bond_type, 1.0)
