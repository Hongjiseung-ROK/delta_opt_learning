"""
학습된 모델로 MMFF 좌표를 DFT-calibrated 좌표로 보정한다.

보정 원리:
  각 결합에 대해 예측 DFT 길이 / MMFF 길이 비율을 구한 후,
  분자 결합 그래프를 DFS로 순회하며 자식 원자 위치를 스케일링한다.
"""
import numpy as np
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import rdGeometry

from delta_chem.ml.feature_extractor import _bond_order, _HYB_MAP

# 모듈 레벨 캐시: 같은 경로의 모델은 한 번만 로드한다
_model_cache: dict[str, object] = {}


def _load_model(model_path: str):
    if model_path not in _model_cache:
        _model_cache[model_path] = joblib.load(model_path)
    return _model_cache[model_path]


def correct_geometry(mol: Chem.Mol, model_path: str) -> Chem.Mol:
    """
    MMFF 최적화된 mol을 받아 bond length 보정이 적용된 새 mol을 반환한다.

    Args:
        mol:        AddHs + MMFFOptimizeMolecule 완료된 RDKit mol
        model_path: 학습된 .joblib 모델 경로

    Returns:
        보정된 좌표를 가진 mol (원본 mol 객체는 변경하지 않음)
    """
    artifact = _load_model(model_path)
    pipe = artifact["pipeline"]
    target_mode = artifact["target_mode"]

    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    ring_info = mol.GetRingInfo()

    # 모든 결합의 feature를 한 번에 수집 → 배치 예측 (N+1 방지)
    rows = []
    bond_indices = []  # (i, j) 쌍 순서 보존
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        elem_pair = sorted([ai.GetSymbol(), aj.GetSymbol()])
        hyb_i = _HYB_MAP.get(ai.GetHybridization(), "OTHER")
        hyb_j = _HYB_MAP.get(aj.GetHybridization(), "OTHER")
        in_ring = bond.IsInRing()
        ring_size = 0
        if in_ring:
            sizes = ring_info.BondRingSizes(bond.GetIdx())
            ring_size = min(sizes) if sizes else 0
        mmff_len = float(np.linalg.norm(coords[i] - coords[j]))

        rows.append({
            "elem1":           elem_pair[0],
            "elem2":           elem_pair[1],
            "bond_order":      _bond_order(bond.GetBondType()),
            "hybridization_1": hyb_i if ai.GetSymbol() == elem_pair[0] else hyb_j,
            "hybridization_2": hyb_j if ai.GetSymbol() == elem_pair[0] else hyb_i,
            "is_in_ring":      int(in_ring),
            "ring_size":       ring_size,
            "mmff_length":     mmff_len,
        })
        bond_indices.append((i, j, mmff_len))

    # 단일 배치 예측
    df_all = pd.DataFrame(rows)
    pred_vals = pipe.predict(df_all)

    # 결합별 스케일 비율 계산
    scale: dict[tuple[int, int], float] = {}
    for (i, j, mmff_len), pred_val in zip(bond_indices, pred_vals):
        if target_mode == "delta":
            pred_len = mmff_len + float(pred_val)
        else:
            pred_len = float(pred_val)
        s = pred_len / mmff_len if mmff_len > 1e-6 else 1.0
        scale[(i, j)] = s
        scale[(j, i)] = s

    # DFS로 결합 그래프를 순회하며 좌표 보정
    new_coords = coords.copy()
    visited = set()
    root = 0
    stack = [root]
    visited.add(root)

    while stack:
        parent = stack.pop()
        for nb in mol.GetAtomWithIdx(parent).GetNeighbors():
            child = nb.GetIdx()
            if child in visited:
                continue
            visited.add(child)
            s = scale.get((parent, child), 1.0)
            vec = new_coords[child] - new_coords[parent]
            new_coords[child] = new_coords[parent] + vec * s
            stack.append(child)

    # 새 Conformer 적용
    new_mol = Chem.RWMol(mol)
    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(new_coords):
        new_conf.SetAtomPosition(i, rdGeometry.Point3D(x, y, z))
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(new_conf, assignId=True)
    return new_mol.GetMol()
