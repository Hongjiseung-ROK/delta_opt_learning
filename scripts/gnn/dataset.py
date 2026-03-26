"""
분자 그래프 데이터셋 빌더

bond_features_combined.csv → PyTorch Geometric Data 객체 리스트

노드(원자) feature:
  - 원소 원-핫 임베딩 인덱스 (elem_idx)
  - hybridization 인덱스 (hyb_idx)

엣지(결합) feature:
  - bond_order, is_in_ring, ring_size (normalized)
  - mmff_length, mmff_angle_1, mmff_angle_2

타겟:
  - dft_length - mmff_length (delta, 결합별)
"""
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch_geometric.data import Data

# ── 원소 / 혼성화 사전 ──────────────────────────────────────────────────────
ELEM_LIST = ["C", "H", "O", "N", "S", "F", "Cl", "Br", "P", "Si", "OTHER"]
HYB_LIST  = ["SP", "SP2", "SP3", "OTHER"]

ELEM_IDX = {e: i for i, e in enumerate(ELEM_LIST)}
HYB_IDX  = {h: i for i, h in enumerate(HYB_LIST)}

NUM_ELEM = len(ELEM_LIST)
NUM_HYB  = len(HYB_LIST)


def _elem_idx(sym: str) -> int:
    return ELEM_IDX.get(sym, ELEM_IDX["OTHER"])


def _hyb_idx(h: str) -> int:
    return HYB_IDX.get(h, HYB_IDX["OTHER"])


def mol_group_to_graph(grp: pd.DataFrame) -> Data | None:
    """
    한 분자의 결합 feature DataFrame을 PyG Data 객체로 변환한다.

    원자 인덱스: bond_idx 컬럼이 아닌 elem1/elem2 + bond_idx를 통해
    암묵적 원자 집합을 재구성한다.

    주의: bond_features CSV에는 원자 인덱스 정보가 없으므로
    결합 목록에서 등장하는 원자를 추론해 노드를 만든다.
    """
    if grp.empty:
        return None

    # ── 원자 집합 재구성 ──────────────────────────────────────────────────
    # bond_idx는 RDKit 결합 번호이며, 원자 번호가 아님.
    # 여기서는 결합 양 끝 원소(elem1, elem2)를 사용해 implicit 노드를 만든다.
    # 정확한 원자 번호가 없으므로: 각 행을 (elem1_node, elem2_node) 쌍으로 취급.
    # bond_idx를 결합 인덱스, 인접 정보를 위해 사용.
    #
    # 실제 분자 그래프 구성을 위해 smiles에서 직접 RDKit으로 원자/결합 읽기.
    try:
        from rdkit import Chem
        smiles = grp["smiles"].iloc[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
    except Exception:
        return None

    n_atoms = mol.GetNumAtoms()
    n_bonds_rdkit = mol.GetNumBonds()

    # ── 노드 feature 벡터 구성 ────────────────────────────────────────────
    # x: [n_atoms, 2]  (elem_idx, hyb_idx)
    # hybridization은 RDKit mol에서 직접 읽어 일관성 보장
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    _HYB_MAP = {
        Chem.rdchem.HybridizationType.SP:  "SP",
        Chem.rdchem.HybridizationType.SP2: "SP2",
        Chem.rdchem.HybridizationType.SP3: "SP3",
    }

    node_elem = []
    node_hyb  = []
    for atom in mol.GetAtoms():
        node_elem.append(_elem_idx(atom.GetSymbol()))
        node_hyb.append(_hyb_idx(_HYB_MAP.get(atom.GetHybridization(), "OTHER")))

    x = torch.tensor(list(zip(node_elem, node_hyb)), dtype=torch.long)  # [N, 2]

    # ── 엣지 feature 및 타겟 구성 ─────────────────────────────────────────
    # CSV의 bond_idx → RDKit bond 객체 매핑
    csv_by_bidx = {row["bond_idx"]: row for _, row in grp.iterrows()}

    edge_index_src = []
    edge_index_dst = []
    edge_attr_list = []
    y_list         = []
    bond_order_ref = []  # 역방향 엣지 쌍 만들 때 참조용

    for bond in mol.GetBonds():
        bidx = bond.GetIdx()
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        if bidx in csv_by_bidx:
            row = csv_by_bidx[bidx]
            bo        = float(row["bond_order"])
            in_ring   = float(row["is_in_ring"])
            ring_size = float(row["ring_size"]) / 10.0   # normalize
            mmff_len  = float(row["mmff_length"])
            ang1      = float(row["mmff_angle_1"]) / 180.0  # normalize
            ang2      = float(row["mmff_angle_2"]) / 180.0
            delta     = float(row["dft_length"]) - mmff_len
        else:
            # H-H 또는 데이터 없는 결합 → 제로 feature, 타겟 0
            bo        = 1.0
            in_ring   = 0.0
            ring_size = 0.0
            mmff_len  = 1.0
            ang1      = 0.0
            ang2      = 0.0
            delta     = 0.0

        ef = [bo, in_ring, ring_size, mmff_len, ang1, ang2]

        # 양방향 엣지 추가 (undirected graph as directed)
        edge_index_src += [i, j]
        edge_index_dst += [j, i]
        edge_attr_list += [ef, ef]
        y_list         += [delta, delta]

    if not edge_index_src:
        return None

    edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)
    y          = torch.tensor(y_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                smiles=smiles, n_atoms=n_atoms)


def build_dataset(
    csv_path: str = "data/features/bond_features_combined.csv",
    exclude: list[str] | None = None,
) -> list[Data]:
    """CSV → PyG Data 리스트"""
    df = pd.read_csv(csv_path)
    if exclude:
        df = df[~df["mol_name"].isin(exclude)]

    graphs = []
    for mol_name, grp in df.groupby("mol_name"):
        g = mol_group_to_graph(grp)
        if g is not None:
            g.mol_name = mol_name
            graphs.append(g)

    print(f"데이터셋: {len(graphs)}개 분자")
    return graphs


if __name__ == "__main__":
    graphs = build_dataset()
    if graphs:
        g = graphs[0]
        print(f"첫 번째 그래프: {g.mol_name}")
        print(f"  노드: {g.x.shape}, 엣지: {g.edge_index.shape}, 타겟: {g.y.shape}")
