"""
FreeSolv_SAMPL.csv 전체 분자에 대해 Gaussian 16 B3LYP/6-31G(d) opt 계산 후
bond feature를 추출하여 bond_features.csv에 추가한다.

사용법:
    # 5개 small test
    conda run -n delta_chem python scripts/collect_freesolv.py --max-mol 5

    # 전체 실행 (기존 bond_features.csv에 append)
    conda run -n delta_chem python scripts/collect_freesolv.py --append

    # Gaussian 계산 없이 feature 추출만 (이미 계산된 경우)
    conda run -n delta_chem python scripts/collect_freesolv.py --extract-only --max-mol 5
"""
import argparse
import re
import tempfile
import os
from pathlib import Path

import pandas as pd

from delta_chem.chem.smiles_to_xyz import smiles_to_xyz
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.chem.gaussian_runner import run_gaussian
from delta_chem.chem.log_parser import parse_log
from delta_chem.ml.feature_extractor import extract_bond_features

FREESOLV_CSV = Path("data/FreeSolv_SAMPL.csv")


def sanitize_name(iupac: str) -> str:
    """IUPAC 이름 → 파일시스템·Gaussian %Chk 안전한 이름으로 변환."""
    name = iupac.strip()
    name = re.sub(r"[,\s/\\()]+", "_", name)  # 쉼표·공백·슬래시·괄호 → _
    name = re.sub(r"[^\w\-]", "", name)        # 나머지 특수문자 제거
    name = re.sub(r"_+", "_", name).strip("_") # 연속 _ 정리
    return name


def load_freesolv(
    max_mol: int | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> list[tuple[str, str]]:
    """FreeSolv CSV → [(sanitized_name, smiles), ...] 반환.

    start_idx, end_idx: CSV 행 기준 슬라이싱 (분산 실행 시 노드별 구간 지정).
    max_mol: 슬라이싱 후 추가 제한 (테스트용).
    """
    df = pd.read_csv(FREESOLV_CSV)
    rows = []
    for _, row in df.iterrows():
        name = sanitize_name(str(row["iupac"]))
        smiles = str(row["smiles"])
        rows.append((name, smiles))
    rows = rows[start_idx:end_idx]
    if max_mol is not None:
        rows = rows[:max_mol]
    return rows


def collect(
    molecules: list[tuple[str, str]],
    output_dir: str = "data/raw",
    route: str = "#p opt B3LYP/6-31G(d)",
    nproc: int = 20,
    mem: str = "56GB",
) -> None:
    """SMILES → .com 생성 → Gaussian 실행. 이미 완료된 분자는 건너뜀."""
    out_dir = Path(output_dir)
    total = len(molecules)
    done, skipped, failed = 0, 0, 0

    for idx, (name, smiles) in enumerate(molecules, 1):
        work_dir = out_dir / name

        # 이미 정상 완료된 경우 건너뜀 (.log 우선, .out 차선)
        existing = None
        for suffix in (".log", ".out"):
            candidate = work_dir / f"{name}_rdkit{suffix}"
            if candidate.exists() and candidate.stat().st_size > 0:
                text = candidate.read_text(encoding="utf-8", errors="replace")
                if "Normal termination" in text:
                    existing = candidate
                    break
        if existing:
            print(f"[{idx:03d}/{total}] {name:40s} — 이미 완료, 건너뜀")
            skipped += 1
            continue

        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx:03d}/{total}] {name:40s} SMILES={smiles}", flush=True)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                rdkit_xyz = os.path.join(tmp, f"{name}_rdkit.xyz")
                smiles_to_xyz(smiles, rdkit_xyz, mol_name=name)

                com_path = str(work_dir / f"{name}_rdkit.com")
                xyz_to_gaussian_com(
                    rdkit_xyz, com_path,
                    route=route, title=name,
                    charge=0, multiplicity=1,
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
    molecules: list[tuple[str, str]],
    output_dir: str = "data/raw",
    output_csv: str = "data/features/bond_features.csv",
    append: bool = False,
) -> pd.DataFrame:
    """Gaussian 출력 → bond feature 추출 → CSV 저장."""
    results_dir = Path(output_dir)
    all_rows = []

    print("\n=== Feature 추출 ===")
    for name, smiles in molecules:
        out_path = None
        for suffix in (".log", ".out"):
            candidate = results_dir / name / f"{name}_rdkit{suffix}"
            if candidate.exists() and candidate.stat().st_size > 0:
                out_path = candidate
                break
        if out_path is None:
            print(f"  [skip] {name}: 출력 파일 없음")
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

    new_df = pd.concat(all_rows, ignore_index=True)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if append and out_path.exists():
        existing = pd.read_csv(out_path)
        # 이미 포함된 분자는 중복 추가하지 않음
        existing_mols = set(existing["mol_name"].unique())
        new_df = new_df[~new_df["mol_name"].isin(existing_mols)]
        if new_df.empty:
            print("추가할 새 분자 없음 (이미 모두 포함됨).")
            return existing
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"\n저장(append): {out_path}  (기존 {len(existing)}행 + 신규 {len(new_df)}행 = {len(combined)}행)")
        return combined
    else:
        new_df.to_csv(out_path, index=False)
        print(f"\n저장: {out_path}  ({len(new_df)}행)")
        return new_df


def main():
    parser = argparse.ArgumentParser(
        description="FreeSolv → Gaussian 계산 → bond_features.csv"
    )
    parser.add_argument("--max-mol", type=int, default=None,
                        help="처리할 최대 분자 수 (기본: 전체, 테스트용)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="FreeSolv CSV 시작 행 인덱스 (분산 실행 시 노드별 구간)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="FreeSolv CSV 끝 행 인덱스 (exclusive)")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--output-csv", default="data/features/bond_features.csv")
    parser.add_argument("--route", default="#p opt B3LYP/6-31G(d)")
    parser.add_argument("--nproc", type=int, default=20)
    parser.add_argument("--mem", default="56GB")
    parser.add_argument("--append", action="store_true",
                        help="기존 bond_features.csv에 append (기본: 새로 작성)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Gaussian 계산 없이 feature 추출만 수행")
    parser.add_argument("--skip-extract", action="store_true",
                        help="feature 추출 없이 Gaussian 계산만 수행 (분산 실행 시 사용)")
    args = parser.parse_args()

    molecules = load_freesolv(args.max_mol, args.start_idx, args.end_idx)
    print(f"처리 대상: {len(molecules)}개 분자 "
          f"(idx {args.start_idx}~{args.end_idx or 'end'})")

    if not args.extract_only:
        collect(molecules, args.output_dir, args.route, args.nproc, args.mem)

    if not args.skip_extract:
        extract(molecules, args.output_dir, args.output_csv, args.append)


if __name__ == "__main__":
    main()
