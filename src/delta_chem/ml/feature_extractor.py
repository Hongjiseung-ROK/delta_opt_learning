"""
MMFF 좌표 + Gaussian DFT 최종 좌표 → bond feature DataFrame

각 결합마다 한 행을 생성한다:
  elem1, elem2, bond_order, hybridization_1, hybridization_2,
  is_in_ring, ring_size, mmff_length → dft_length  (학습 타겟)

MMFF 좌표는 Gaussian .out의 'Input orientation:' 블록(첫 번째 occurrence)에서
직접 파싱한다. ETKDGv3를 재실행하면 난수 시드에 따라 다른 conformer가 생성될 수
있어 Gaussian에 실제로 제출된 구조와 불일치가 발생한다.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from delta_chem.chem.log_parser import parse_input_geometry, parse_final_geometry


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

    MMFF 좌표: Gaussian .out의 Input orientation(첫 번째 블록) 사용
    DFT 좌표:  Gaussian .out의 Standard orientation(마지막 블록) 사용

    원자 인덱스 정렬 보장:
    - smiles_to_xyz()가 RDKit mol 원자 순서로 XYZ를 씀
    - gaussian_writer.py가 동일 순서로 .com을 생성
    - Gaussian의 Center Number = 해당 순서 그대로
    → RDKit mol.GetAtomWithIdx(i) ↔ Gaussian Center i+1
    """
    # 1. RDKit mol (결합 위상 정보만 필요, 3D 좌표는 .out에서 읽음)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.DataFrame()
    mol = Chem.AddHs(mol)

    # 2. MMFF 초기 좌표: Input orientation (Gaussian에 실제 제출된 구조)
    mmff_atoms = parse_input_geometry(gaussian_out_path)
    if not mmff_atoms or len(mmff_atoms) != mol.GetNumAtoms():
        return pd.DataFrame()
    mmff_coords = np.array([[x, y, z] for _, x, y, z in mmff_atoms])

    # 3. DFT 최종 좌표: Standard orientation (마지막 블록)
    dft_atoms = parse_final_geometry(gaussian_out_path)
    if not dft_atoms or len(dft_atoms) != mol.GetNumAtoms():
        return pd.DataFrame()
    dft_coords = np.array([[x, y, z] for _, x, y, z in dft_atoms])

    # 4. 혼성화 정보: ETKDGv3로 임시 conformer 생성 (위상/혼성화 파악용만)
    mol_3d = Chem.RWMol(mol)
    AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol_3d)

    ring_info = mol_3d.GetRingInfo()

    rows = []
    for bond in mol_3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom_i = mol_3d.GetAtomWithIdx(i)
        atom_j = mol_3d.GetAtomWithIdx(j)

        elem_pair = sorted([atom_i.GetSymbol(), atom_j.GetSymbol()])
        bond_order = _bond_order(bond.GetBondType())
        hyb_i = _HYB_MAP.get(atom_i.GetHybridization(), "OTHER")
        hyb_j = _HYB_MAP.get(atom_j.GetHybridization(), "OTHER")

        in_ring = bond.IsInRing()
        ring_size = 0
        if in_ring:
            sizes = ring_info.BondRingSizes(bond.GetIdx())
            ring_size = min(sizes) if sizes else 0

        # 좌표는 .out에서 파싱한 값 사용
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
    condition: str = "rdkit",
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
