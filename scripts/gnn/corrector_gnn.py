"""
학습된 GNN 모델로 MMFF 좌표를 보정한다.

기존 corrector.py의 DFS 방식 대신
RDKit AllChem.SetBondLength()를 사용하여 구조 왜곡을 최소화한다.

SetBondLength 방식:
  - RDKit 내장 함수가 결합 길이를 직접 설정
  - 결합 j에 붙은 모든 원자들이 rigid body로 이동 (각도 보존 시도)
  - DFS 스케일링보다 결합각 보존 성능 우수
"""
import sys
import os
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

from scripts.gnn.dataset import _elem_idx, _hyb_idx, NUM_ELEM, NUM_HYB
from scripts.gnn.model import BondLengthGNN

# 모듈 레벨 캐시
_gnn_cache: dict[str, object] = {}


def _load_gnn(model_path: str) -> tuple[BondLengthGNN, str]:
    if model_path not in _gnn_cache:
        artifact = joblib.load(model_path)
        kwargs = artifact.get("model_kwargs", {})
        model = BondLengthGNN(**kwargs)
        model.load_state_dict(artifact["model_state"])
        model.eval()
        _gnn_cache[model_path] = (model, artifact.get("target_mode", "delta"))
    return _gnn_cache[model_path]


def _build_graph_tensors(mol: Chem.Mol, coords: np.ndarray):
    """RDKit mol + coords → (x, edge_index, edge_attr, bond_idx_list)"""
    from delta_chem.ml.feature_extractor import _mean_angle_at, _bond_order, _HYB_MAP

    ring_info = mol.GetRingInfo()
    n_atoms = mol.GetNumAtoms()

    # 노드 feature
    node_elem = []
    node_hyb  = []
    for atom in mol.GetAtoms():
        node_elem.append(_elem_idx(atom.GetSymbol()))
        node_hyb.append(_hyb_idx(_HYB_MAP.get(atom.GetHybridization(), "OTHER")))
    x = torch.tensor(list(zip(node_elem, node_hyb)), dtype=torch.long)

    # 엣지 feature
    edge_src, edge_dst = [], []
    edge_attr_list = []
    bond_info = []  # (i, j, mmff_len) for result mapping

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bo = _bond_order(bond.GetBondType())
        in_ring = int(bond.IsInRing())
        ring_size = 0
        if in_ring:
            sizes = ring_info.BondRingSizes(bond.GetIdx())
            ring_size = min(sizes) if sizes else 0
        mmff_len = float(np.linalg.norm(coords[i] - coords[j]))
        ang_i = _mean_angle_at(i, j, coords, mol)
        ang_j = _mean_angle_at(j, i, coords, mol)

        ai = mol.GetAtomWithIdx(i)
        aj = mol.GetAtomWithIdx(j)
        elem_pair = sorted([ai.GetSymbol(), aj.GetSymbol()])
        if ai.GetSymbol() == elem_pair[0]:
            ang1, ang2 = ang_i, ang_j
        else:
            ang1, ang2 = ang_j, ang_i

        ef = [bo, float(in_ring), ring_size / 10.0, mmff_len, ang1 / 180.0, ang2 / 180.0]

        edge_src += [i, j]
        edge_dst += [j, i]
        edge_attr_list += [ef, ef]
        bond_info.append((i, j, mmff_len, bond.GetIdx()))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)

    return x, edge_index, edge_attr, bond_info


def correct_geometry_gnn(mol: Chem.Mol, model_path: str) -> Chem.Mol:
    """
    GNN으로 MMFF 좌표를 DFT-calibrated 좌표로 보정.

    Args:
        mol:        AddHs + MMFFOptimizeMolecule 완료된 RDKit mol
        model_path: 학습된 GNN .joblib 경로

    Returns:
        보정된 좌표를 가진 mol (원본 변경 없음)
    """
    model, target_mode = _load_gnn(model_path)

    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    x, edge_index, edge_attr, bond_info = _build_graph_tensors(mol, coords)

    with torch.no_grad():
        pred_per_edge = model(x, edge_index, edge_attr)  # [2*n_bonds]

    # 양방향 엣지 중 순방향(짝수 인덱스)만 사용
    pred_per_bond = pred_per_edge[0::2].numpy()  # [n_bonds]

    # 예측된 결합 길이 계산
    pred_lengths = {}
    for (i, j, mmff_len, bidx), delta in zip(bond_info, pred_per_bond):
        if target_mode == "delta":
            pred_len = mmff_len + float(delta)
        else:
            pred_len = float(delta)
        pred_lengths[bidx] = (i, j, max(pred_len, 0.5))  # 물리적 하한

    # ── SetBondLength로 보정 (DFS 대신) ───────────────────────────────────
    new_mol = Chem.RWMol(mol)
    # conformer를 복사하여 작업
    new_mol.RemoveAllConformers()
    from rdkit.Geometry import rdGeometry
    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x_, y_, z_) in enumerate(coords):
        new_conf.SetAtomPosition(i, rdGeometry.Point3D(x_, y_, z_))
    new_mol.AddConformer(new_conf, assignId=True)

    # spanning tree 순서로 결합 길이 설정 (각도 보존 극대화)
    # BFS로 spanning tree 구성
    from collections import deque
    adj = {a.GetIdx(): [] for a in new_mol.GetAtoms()}
    for bond in new_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append((j, bond.GetIdx()))
        adj[j].append((i, bond.GetIdx()))

    visited = set()
    queue = deque([0])
    visited.add(0)
    traversal_order = []  # (parent, child, bond_idx)

    while queue:
        parent = queue.popleft()
        for child, bidx in adj[parent]:
            if child not in visited:
                visited.add(child)
                traversal_order.append((parent, child, bidx))
                queue.append(child)

    # BFS 순서로 SetBondLength 적용
    conf_new = new_mol.GetConformer()
    for parent, child, bidx in traversal_order:
        if bidx in pred_lengths:
            _, _, pred_len = pred_lengths[bidx]
            try:
                rdMolTransforms.SetBondLength(conf_new, parent, child, pred_len)
            except Exception:
                pass  # ring closure 등 예외 무시

    mol_result = new_mol.GetMol()
    return mol_result


if __name__ == "__main__":
    # 간단한 동작 테스트
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles("CC(=O)O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    model_path = "models/bond_length_corrector_gnn.joblib"
    if Path(model_path).exists():
        result = correct_geometry_gnn(mol, model_path)
        print(f"보정 완료. 원자 수: {result.GetNumAtoms()}")
    else:
        print(f"모델 없음: {model_path}")
