"""
FreeSolv 데이터셋에서 선택한 5개 분자에 대해
Gaussian 16 B3LYP/6-31G(d) opt 계산 실행 + bond feature 추출.

사용법:
    conda run -n delta_chem python scripts/collect_freesolv5.py
    conda run -n delta_chem python scripts/collect_freesolv5.py --nproc 20 --mem 40GB
"""
import argparse
import tempfile
import os
from pathlib import Path

from delta_chem.chem.smiles_to_xyz import smiles_to_xyz
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.chem.gaussian_runner import run_gaussian
from delta_chem.chem.log_parser import parse_log
from delta_chem.ml.feature_extractor import build_dataset

# FreeSolv에서 선정한 5개 분자 (기존 훈련셋 미포함, C/H/O만 사용)
MOLECULES = [
    # (name,                     smiles,              charge, mult)
    # 이름에 쉼표 불가 — Gaussian %Chk 파싱 오류 발생
    ("3-methylbut-1-ene",        "C=CC(C)C",          0, 1),
    ("2_3-dimethylbutane",       "CC(C)C(C)C",        0, 1),
    ("2-methylpentan-2-ol",      "CCCC(C)(C)O",       0, 1),
    ("cycloheptanol",            "OC1CCCCCC1",        0, 1),
    ("2_4-dimethylpentan-3-one", "CC(C)C(=O)C(C)C",  0, 1),
]


def collect(
    output_dir: str = "data/raw",
    route: str = "#p opt B3LYP/6-31G(d)",
    nproc: int = 20,
    mem: str = "40GB",
) -> None:
    out_dir = Path(output_dir)
    total = len(MOLECULES)
    done, skipped, failed = 0, 0, 0

    for idx, (name, smiles, charge, mult) in enumerate(MOLECULES, 1):
        work_dir = out_dir / name
        out_path = work_dir / f"{name}_rdkit.out"

        # 이미 정상 완료된 경우 건너뜀
        if out_path.exists():
            text = out_path.read_text(encoding="utf-8", errors="replace")
            if "Normal termination" in text:
                print(f"[{idx}/{total}] {name:30s} — 이미 완료, 건너뜀")
                skipped += 1
                continue

        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{total}] {name:30s} SMILES={smiles}", flush=True)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                rdkit_xyz = os.path.join(tmp, f"{name}_rdkit.xyz")
                smiles_to_xyz(smiles, rdkit_xyz, mol_name=name)

                com_path = str(work_dir / f"{name}_rdkit.com")
                xyz_to_gaussian_com(
                    rdkit_xyz, com_path,
                    route=route, title=name,
                    charge=charge, multiplicity=mult,
                    nproc=nproc, mem=mem,
                )

            result_out = run_gaussian(com_path)
            res = parse_log(result_out)

            status = "✓" if res.normal_termination else "✗ (비수렴)"
            print(f"          → steps={res.opt_steps:3d}  cpu={res.cpu_seconds:6.1f}s  {status}")
            done += 1

        except Exception as e:
            print(f"          → 실패: {e}")
            failed += 1

    print(f"\n완료: {done}개 성공 / {skipped}개 건너뜀 / {failed}개 실패 (총 {total}개)")


def extract(
    output_dir: str = "data/raw",
    output_csv: str = "data/features/freesolv5_bond_features.csv",
) -> None:
    mol_list = [(name, smiles) for name, smiles, *_ in MOLECULES]
    print("\n=== Feature 추출 ===")
    df = build_dataset(
        results_dir=output_dir,
        molecule_list=mol_list,
        condition="rdkit",
        output_csv=output_csv,
    )
    if df.empty:
        print("feature 추출 실패")
        return
    print(f"\n--- 추출 결과 ---")
    print(df.groupby(["elem1", "elem2", "bond_order"])[["mmff_length", "dft_length"]].mean().round(4))
    print(f"\nCSV 저장: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--route", default="#p opt B3LYP/6-31G(d)")
    parser.add_argument("--nproc", type=int, default=20)
    parser.add_argument("--mem", default="40GB")
    parser.add_argument("--output-csv", default="data/features/freesolv5_bond_features.csv")
    parser.add_argument("--extract-only", action="store_true",
                        help="Gaussian 계산 없이 feature 추출만 수행")
    args = parser.parse_args()

    if not args.extract_only:
        collect(args.output_dir, args.route, args.nproc, args.mem)

    extract(args.output_dir, args.output_csv)


if __name__ == "__main__":
    main()
