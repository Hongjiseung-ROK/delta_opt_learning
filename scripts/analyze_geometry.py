"""
MMFF vs DFT 기하 구조 비교 분석 (결합각 / 이면각)

data/raw/ 내 계산 완료 분자를 전수 분석하여 아래를 보고한다:
  1. 결합각(bond angle)  MMFF vs DFT 오차 분포
  2. 이면각(dihedral)    MMFF vs DFT 오차 분포
  3. Conformer 전환(|Δdihedral| > 30°) 비율
  4. ML 학습 가능성 평가 (체계적 오차 vs 무작위 오차)

사용법:
    conda run -n delta_chem python scripts/analyze_geometry.py
    conda run -n delta_chem python scripts/analyze_geometry.py --conf-thresh 30
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(__file__))
from delta_chem.chem.log_parser import parse_input_geometry, parse_final_geometry

# ── 기하 계산 유틸리티 ──────────────────────────────────────────────────────────

def calc_bond_angle(p1, p2, p3):
    """∠p1-p2-p3 (도). p2가 꼭짓점."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def calc_dihedral(p1, p2, p3, p4):
    """이면각 ∠p1-p2-p3-p4 (도, -180~180)."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    norm_b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    m1 = np.cross(n1, norm_b2)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.degrees(np.arctan2(y, x))


def circular_diff(a, b):
    """두 각도(도)의 최단 차이 (-180~180)."""
    d = (a - b + 180) % 360 - 180
    return d


# ── 분자별 특징 추출 ────────────────────────────────────────────────────────────

def get_topology(smiles):
    """RDKit mol 반환 (H 포함, 3D conformer 없이 위상만)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    # 혼성화 정보만을 위한 임시 embed
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def extract_angles(mol, coords):
    """모든 결합각 반환: [(label, angle_deg), ...]"""
    rows = []
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
        if len(neighbors) < 2:
            continue
        elem_j = atom.GetSymbol()
        for i in range(len(neighbors)):
            for k in range(i + 1, len(neighbors)):
                ni, nk = neighbors[i], neighbors[k]
                elem_i = mol.GetAtomWithIdx(ni).GetSymbol()
                elem_k = mol.GetAtomWithIdx(nk).GetSymbol()
                label = "-".join(sorted([elem_i, elem_k])) + f"_{elem_j}"
                angle = calc_bond_angle(coords[ni], coords[j], coords[nk])
                rows.append({"label": label, "center": elem_j,
                             "left": elem_i, "right": elem_k, "angle": angle})
    return rows


def extract_dihedrals(mol, coords):
    """
    회전 가능한 결합(rotatable bond) 기준 이면각 반환.
    각 결합에 대해 heavy-atom 이웃의 모든 조합을 열거한다.
    """
    rows = []
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        atom_j = mol.GetAtomWithIdx(j)
        atom_k = mol.GetAtomWithIdx(k)

        # 단일결합만 (1.0 또는 방향족 아닌 것)
        bt = bond.GetBondTypeAsDouble()
        if bt > 1.0:
            continue

        # 양끝 원자 모두에 이웃이 있어야 함 (terminal 제외)
        neighbors_j = [nb.GetIdx() for nb in atom_j.GetNeighbors() if nb.GetIdx() != k]
        neighbors_k = [nb.GetIdx() for nb in atom_k.GetNeighbors() if nb.GetIdx() != j]
        if not neighbors_j or not neighbors_k:
            continue

        # heavy-atom 이웃 우선, 없으면 H 포함
        heavy_j = [n for n in neighbors_j if mol.GetAtomWithIdx(n).GetSymbol() != "H"]
        heavy_k = [n for n in neighbors_k if mol.GetAtomWithIdx(n).GetSymbol() != "H"]
        sel_j = heavy_j if heavy_j else neighbors_j[:1]
        sel_k = heavy_k if heavy_k else neighbors_k[:1]

        elem_j = atom_j.GetSymbol()
        elem_k = atom_k.GetSymbol()
        in_ring = bond.IsInRing()

        for i in sel_j:
            for l in sel_k:
                elem_i = mol.GetAtomWithIdx(i).GetSymbol()
                elem_l = mol.GetAtomWithIdx(l).GetSymbol()
                # 정규화된 레이블 (j-k 방향 고정)
                label = f"{elem_i}-{elem_j}-{elem_k}-{elem_l}"
                central = f"{'-'.join(sorted([elem_j, elem_k]))}"
                dihedral = calc_dihedral(coords[i], coords[j], coords[k], coords[l])
                rows.append({
                    "label": label,
                    "central_bond": central,
                    "elem_j": elem_j,
                    "elem_k": elem_k,
                    "in_ring": int(in_ring),
                    "dihedral": dihedral,
                })
    return rows


# ── 메인 분석 ──────────────────────────────────────────────────────────────────

def load_molecules(results_dir: Path):
    """
    data/raw/ 하위 디렉토리를 스캔하여 Normal termination인 분자 목록 반환.
    bond_features.csv에서 smiles를 조회한다.
    """
    feat_csv = Path("data/features/bond_features.csv")
    smiles_map = {}
    if feat_csv.exists():
        df = pd.read_csv(feat_csv)
        for _, row in df.drop_duplicates("mol_name").iterrows():
            smiles_map[row["mol_name"]] = row["smiles"]

    # FreeSolv test 분자도 추가
    for csv in [Path("data/features/freesolv_test_bond_features.csv"),
                Path("data/features/freesolv_bond_features.csv")]:
        if csv.exists():
            df2 = pd.read_csv(csv)
            for _, row in df2.drop_duplicates("mol_name").iterrows():
                smiles_map.setdefault(row["mol_name"], row["smiles"])

    mols = []
    for mol_dir in sorted(results_dir.iterdir()):
        if not mol_dir.is_dir():
            continue
        name = mol_dir.name
        if name not in smiles_map:
            continue
        log = None
        for suffix in (".log", ".out"):
            c = mol_dir / f"{name}_rdkit{suffix}"
            if c.exists() and c.stat().st_size > 0:
                text = c.read_text(encoding="utf-8", errors="replace")
                if "Normal termination" in text:
                    log = c
                    break
        if log:
            mols.append((name, smiles_map[name], str(log)))
    return mols


def analyze(results_dir="data/raw", conf_thresh=30.0, output_dir="figures"):
    Path(output_dir).mkdir(exist_ok=True)
    mols = load_molecules(Path(results_dir))
    print(f"분석 대상: {len(mols)}개 분자\n")

    angle_rows = []     # 결합각
    dihedral_rows = []  # 이면각

    for name, smiles, log_path in mols:
        mol = get_topology(smiles)
        if mol is None:
            print(f"  [skip] {name}: SMILES 파싱 실패")
            continue

        mmff_atoms = parse_input_geometry(log_path)
        dft_atoms  = parse_final_geometry(log_path)
        if not mmff_atoms or not dft_atoms:
            print(f"  [skip] {name}: 좌표 파싱 실패")
            continue
        if len(mmff_atoms) != mol.GetNumAtoms() or len(dft_atoms) != mol.GetNumAtoms():
            print(f"  [skip] {name}: 원자 수 불일치")
            continue

        mmff_coords = np.array([[x, y, z] for _, x, y, z in mmff_atoms])
        dft_coords  = np.array([[x, y, z] for _, x, y, z in dft_atoms])

        # 결합각
        mmff_angles = {(r["label"], i): r["angle"]
                       for i, r in enumerate(extract_angles(mol, mmff_coords))}
        for i, r in enumerate(extract_angles(mol, dft_coords)):
            key = (r["label"], i)
            if key in mmff_angles:
                delta = r["angle"] - mmff_angles[key]
                angle_rows.append({
                    "mol_name": name,
                    "label": r["label"],
                    "center": r["center"],
                    "mmff_angle": mmff_angles[key],
                    "dft_angle": r["angle"],
                    "delta": delta,
                })

        # 이면각
        mmff_dihedrals = {(r["label"], i): r["dihedral"]
                          for i, r in enumerate(extract_dihedrals(mol, mmff_coords))}
        for i, r in enumerate(extract_dihedrals(mol, dft_coords)):
            key = (r["label"], i)
            if key in mmff_dihedrals:
                delta = circular_diff(r["dihedral"], mmff_dihedrals[key])
                dihedral_rows.append({
                    "mol_name": name,
                    "label": r["label"],
                    "central_bond": r["central_bond"],
                    "elem_j": r["elem_j"],
                    "elem_k": r["elem_k"],
                    "in_ring": r["in_ring"],
                    "mmff_dihedral": mmff_dihedrals[key],
                    "dft_dihedral": r["dihedral"],
                    "delta": delta,
                })

        print(f"  [ok] {name}: {len(angle_rows)}개 결합각 / {len(dihedral_rows)}개 이면각 누적")

    if not angle_rows or not dihedral_rows:
        print("데이터 부족 — 분석 중단")
        return

    df_ang = pd.DataFrame(angle_rows)
    df_dih = pd.DataFrame(dihedral_rows)
    df_dih["abs_delta"] = df_dih["delta"].abs()
    df_dih["conformer_change"] = df_dih["abs_delta"] > conf_thresh

    # CSV 저장
    df_ang.to_csv(f"{output_dir}/08_bond_angle_deltas.csv", index=False)
    df_dih.to_csv(f"{output_dir}/08_dihedral_deltas.csv", index=False)

    # ── 통계 보고 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("결합각(Bond Angle) 오차 통계")
    print("=" * 60)
    print(f"  전체 결합각 수   : {len(df_ang):,}")
    print(f"  |Δangle| 평균   : {df_ang['delta'].abs().mean():.3f}°")
    print(f"  |Δangle| 중앙값 : {df_ang['delta'].abs().median():.3f}°")
    print(f"  |Δangle| 최대   : {df_ang['delta'].abs().max():.3f}°")
    print(f"  |Δangle| > 5°   : {(df_ang['delta'].abs() > 5).sum()}개 "
          f"({(df_ang['delta'].abs() > 5).mean()*100:.1f}%)")

    print("\n중심 원자별 결합각 오차 (|Δ| 평균):")
    center_stats = df_ang.groupby("center")["delta"].agg(
        count="count", mean_abs=lambda x: x.abs().mean(), std="std"
    ).sort_values("mean_abs", ascending=False)
    print(center_stats.round(3).to_string())

    print("\n" + "=" * 60)
    print("이면각(Dihedral Angle) 오차 통계")
    print("=" * 60)
    print(f"  전체 이면각 수         : {len(df_dih):,}")
    print(f"  |Δdihedral| 평균      : {df_dih['abs_delta'].mean():.2f}°")
    print(f"  |Δdihedral| 중앙값    : {df_dih['abs_delta'].median():.2f}°")
    print(f"  |Δdihedral| > 30°    : {df_dih['conformer_change'].sum()}개 "
          f"({df_dih['conformer_change'].mean()*100:.1f}%) ← conformer 전환 의심")
    print(f"  |Δdihedral| ≤ 30°    : {(~df_dih['conformer_change']).sum()}개 "
          f"({(~df_dih['conformer_change']).mean()*100:.1f}%) ← 학습 가능 후보")

    small = df_dih[~df_dih["conformer_change"]]
    print(f"\n  [conformer 전환 제외 후]")
    print(f"  |Δdihedral| 평균      : {small['abs_delta'].mean():.2f}°")
    print(f"  |Δdihedral| 중앙값    : {small['abs_delta'].median():.2f}°")
    print(f"  |Δdihedral| > 10°    : {(small['abs_delta'] > 10).sum()}개 "
          f"({(small['abs_delta'] > 10).mean()*100:.1f}%)")

    print("\n중심 결합 유형별 이면각 오차 (|Δ| 평균, 건수 10개 이상):")
    bond_stats = df_dih.groupby("central_bond").agg(
        count=("delta", "count"),
        mean_abs=("abs_delta", "mean"),
        std=("abs_delta", "std"),
        conf_change_pct=("conformer_change", lambda x: x.mean() * 100),
    ).query("count >= 10").sort_values("mean_abs", ascending=False)
    print(bond_stats.round(2).to_string())

    # ── 그래프 ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("MMFF vs DFT 기하 구조 비교 분석", fontsize=14)

    # (0,0) 결합각 Δ 분포
    ax = axes[0, 0]
    ax.hist(df_ang["delta"], bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ Bond Angle (°)")
    ax.set_ylabel("Count")
    ax.set_title(f"결합각 오차 분포\n"
                 f"MAE={df_ang['delta'].abs().mean():.2f}°, "
                 f"std={df_ang['delta'].std():.2f}°")

    # (0,1) 중심 원자별 결합각 오차 박스플롯
    ax = axes[0, 1]
    centers = df_ang.groupby("center")["delta"].apply(list)
    ax.boxplot(centers.values, labels=centers.index, showfliers=False)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("중심 원자")
    ax.set_ylabel("Δ Bond Angle (°)")
    ax.set_title("중심 원자별 결합각 오차")

    # (0,2) MMFF vs DFT 결합각 scatter
    ax = axes[0, 2]
    ax.scatter(df_ang["mmff_angle"], df_ang["dft_angle"], alpha=0.3, s=5, color="steelblue")
    lim = [df_ang[["mmff_angle","dft_angle"]].min().min() - 2,
           df_ang[["mmff_angle","dft_angle"]].max().max() + 2]
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("MMFF Bond Angle (°)")
    ax.set_ylabel("DFT Bond Angle (°)")
    ax.set_title("결합각 MMFF vs DFT")

    # (1,0) 이면각 Δ 전체 분포 (conformer 전환 구분)
    ax = axes[1, 0]
    ok  = df_dih[~df_dih["conformer_change"]]["delta"]
    bad = df_dih[ df_dih["conformer_change"]]["delta"]
    ax.hist(ok,  bins=72, color="steelblue", alpha=0.7, label=f"|Δ|≤{conf_thresh:.0f}° ({len(ok)}개)")
    ax.hist(bad, bins=72, color="tomato",    alpha=0.7, label=f"|Δ|>{conf_thresh:.0f}° ({len(bad)}개)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ Dihedral (°)")
    ax.set_ylabel("Count")
    ax.set_title(f"이면각 오차 분포\n(conf_thresh={conf_thresh:.0f}°)")
    ax.legend(fontsize=8)

    # (1,1) 중심 결합 유형별 이면각 오차 박스플롯 (10건 이상)
    ax = axes[1, 1]
    valid_bonds = bond_stats.index.tolist()
    bond_data = [df_dih[df_dih["central_bond"] == b]["delta"].values
                 for b in valid_bonds]
    ax.boxplot(bond_data, labels=valid_bonds, showfliers=False)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("중심 결합 유형")
    ax.set_ylabel("Δ Dihedral (°)")
    ax.set_title("결합 유형별 이면각 오차")
    ax.tick_params(axis="x", rotation=45)

    # (1,2) MMFF vs DFT 이면각 scatter (conformer 전환 구분)
    ax = axes[1, 2]
    ok_df  = df_dih[~df_dih["conformer_change"]]
    bad_df = df_dih[ df_dih["conformer_change"]]
    ax.scatter(ok_df["mmff_dihedral"],  ok_df["dft_dihedral"],
               alpha=0.3, s=4, color="steelblue", label=f"same conformer ({len(ok_df)})")
    ax.scatter(bad_df["mmff_dihedral"], bad_df["dft_dihedral"],
               alpha=0.5, s=6, color="tomato",    label=f"conformer change ({len(bad_df)})")
    ax.plot([-180, 180], [-180, 180], "k--", linewidth=1)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("MMFF Dihedral (°)")
    ax.set_ylabel("DFT Dihedral (°)")
    ax.set_title("이면각 MMFF vs DFT")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = f"{output_dir}/08_geometry_analysis.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n그래프 저장: {fig_path}")
    print(f"CSV 저장  : {output_dir}/08_bond_angle_deltas.csv")
    print(f"           {output_dir}/08_dihedral_deltas.csv")

    # ── ML 가능성 평가 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ML 학습 가능성 평가")
    print("=" * 60)

    ang_mae = df_ang["delta"].abs().mean()
    dih_mae_all  = df_dih["abs_delta"].mean()
    dih_mae_filt = small["abs_delta"].mean()
    conf_pct = df_dih["conformer_change"].mean() * 100

    print(f"\n  결합각:")
    print(f"    MAE={ang_mae:.2f}°  → ", end="")
    if ang_mae < 2.0:
        print("체계적 오차 작음, 학습 가능성 높음 ✓")
    elif ang_mae < 5.0:
        print("중간 수준 오차, 학습 시 개선 여지 있음 △")
    else:
        print("오차 큼, 학습 필요 !")

    print(f"\n  이면각 (전체):")
    print(f"    MAE={dih_mae_all:.2f}°, conformer 전환 {conf_pct:.1f}%")
    print(f"  이면각 (conformer 전환 제외):")
    print(f"    MAE={dih_mae_filt:.2f}°  → ", end="")
    if dih_mae_filt < 5.0:
        print("체계적 오차 작음, 학습 가능성 있음 ✓")
    elif dih_mae_filt < 15.0:
        print("중간 수준 오차, sin/cos 인코딩 + 필터링 필요 △")
    else:
        print("오차 큼 or 무작위성 높음 → 학습 어려움 ✗")

    if conf_pct > 20:
        print(f"\n  ⚠ conformer 전환 비율({conf_pct:.1f}%)이 높음")
        print(f"    → 동일 conformer 내 보정만 학습해도 효과 제한적일 수 있음")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/raw")
    parser.add_argument("--conf-thresh", type=float, default=30.0,
                        help="conformer 전환 판정 임계값 (기본: 30도)")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    analyze(args.results_dir, args.conf_thresh, args.output_dir)


if __name__ == "__main__":
    main()
