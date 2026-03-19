"""
다양한 유기 분자 50개에 대해 RDKit → Gaussian B3LYP/6-31G(d) opt 실행.
이미 완료된 분자는 건너뜀.

사용법:
    conda run -n delta_chem python scripts/collect_data.py
    conda run -n delta_chem python scripts/collect_data.py --nproc 4 --mem 4GB
"""
import argparse
from pathlib import Path

from delta_chem.chem.smiles_to_xyz import smiles_to_xyz
from delta_chem.chem.gaussian_writer import xyz_to_gaussian_com
from delta_chem.chem.gaussian_runner import run_gaussian
from delta_chem.chem.log_parser import parse_log

# fmt: off
MOLECULES = [
    # (name, SMILES, charge, mult)
    # ── 알케인 ──────────────────────────────────────────────
    ("methane",         "C",                0, 1),
    ("ethane",          "CC",               0, 1),
    ("propane",         "CCC",              0, 1),
    ("butane",          "CCCC",             0, 1),
    ("pentane",         "CCCCC",            0, 1),
    ("hexane",          "CCCCCC",           0, 1),
    ("neopentane",      "CC(C)(C)C",        0, 1),
    ("cyclohexane",     "C1CCCCC1",         0, 1),
    # ── 알켄 / 알카인 ────────────────────────────────────────
    ("ethylene",        "C=C",              0, 1),
    ("propylene",       "CC=C",             0, 1),
    ("isobutylene",     "CC(=C)C",          0, 1),
    ("acetylene",       "C#C",              0, 1),
    ("propyne",         "CC#C",             0, 1),
    # ── 방향족 ──────────────────────────────────────────────
    ("benzene",         "c1ccccc1",         0, 1),
    ("toluene",         "Cc1ccccc1",        0, 1),
    ("naphthalene",     "c1ccc2ccccc2c1",   0, 1),
    # ── 헤테로고리 ──────────────────────────────────────────
    ("pyridine",        "c1ccncc1",         0, 1),
    ("furan",           "c1ccoc1",          0, 1),
    ("thiophene",       "c1ccsc1",          0, 1),
    ("pyrrole",         "c1cc[nH]c1",       0, 1),
    ("piperidine",      "C1CCNCC1",         0, 1),
    ("morpholine",      "C1COCCN1",         0, 1),
    ("THF",             "C1CCOC1",          0, 1),
    ("dioxane",         "C1COCCO1",         0, 1),
    # ── 알코올 ──────────────────────────────────────────────
    ("methanol",        "CO",               0, 1),
    ("ethanol",         "CCO",              0, 1),
    ("1-propanol",      "CCCO",             0, 1),
    ("2-propanol",      "CC(O)C",           0, 1),
    ("tert-butanol",    "CC(C)(C)O",        0, 1),
    ("cyclohexanol",    "OC1CCCCC1",        0, 1),
    ("phenol",          "Oc1ccccc1",        0, 1),
    # ── 에터 ────────────────────────────────────────────────
    ("dimethyl_ether",  "COC",              0, 1),
    ("diethyl_ether",   "CCOCC",            0, 1),
    # ── 알데히드 / 케톤 ──────────────────────────────────────
    ("formaldehyde",    "C=O",              0, 1),
    ("acetaldehyde",    "CC=O",             0, 1),
    ("acetone",         "CC(C)=O",          0, 1),
    ("MEK",             "CCC(C)=O",         0, 1),
    ("cyclohexanone",   "O=C1CCCCC1",       0, 1),
    # ── 카르복실산 / 에스터 ───────────────────────────────────
    ("formic_acid",     "OC=O",             0, 1),
    ("acetic_acid",     "CC(=O)O",          0, 1),
    ("propionic_acid",  "CCC(=O)O",         0, 1),
    ("methyl_acetate",  "COC(C)=O",         0, 1),
    ("ethyl_acetate",   "CCOC(C)=O",        0, 1),
    # ── 아민 ────────────────────────────────────────────────
    ("methylamine",     "CN",               0, 1),
    ("dimethylamine",   "CNC",              0, 1),
    ("trimethylamine",  "CN(C)C",           0, 1),
    ("aniline",         "Nc1ccccc1",        0, 1),
    # ── 기타 ────────────────────────────────────────────────
    ("acetonitrile",    "CC#N",             0, 1),
    ("acetamide",       "CC(N)=O",          0, 1),
    ("chloromethane",   "CCl",              0, 1),
]
# fmt: on

assert len(MOLECULES) == 50, f"분자 수 확인: {len(MOLECULES)}"


def collect(
    output_dir: str = "data/raw",
    route: str = "#p opt B3LYP/6-31G(d)",
    nproc: int = 4,
    mem: str = "4GB",
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
                print(f"[{idx:02d}/{total}] {name:20s} — 이미 완료, 건너뜀")
                skipped += 1
                continue

        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx:02d}/{total}] {name:20s} SMILES={smiles}", flush=True)

        try:
            # RDKit → XYZ
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmp:
                rdkit_xyz = os.path.join(tmp, f"{name}_rdkit.xyz")
                smiles_to_xyz(smiles, rdkit_xyz, mol_name=name)

                # Gaussian .com 생성
                com_path = str(work_dir / f"{name}_rdkit.com")
                xyz_to_gaussian_com(
                    rdkit_xyz, com_path,
                    route=route, title=name,
                    charge=charge, multiplicity=mult,
                    nproc=nproc, mem=mem,
                )

            # Gaussian 실행
            result_out = run_gaussian(com_path)
            res = parse_log(result_out)

            status = "✓" if res.normal_termination else "✗ (비수렴)"
            print(f"          → steps={res.opt_steps:3d}  cpu={res.cpu_seconds:6.1f}s  {status}")
            done += 1

        except Exception as e:
            print(f"          → 실패: {e}")
            failed += 1

    print(f"\n완료: {done}개 성공 / {skipped}개 건너뜀 / {failed}개 실패 (총 {total}개)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--route", default="#p opt B3LYP/6-31G(d)")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--mem", default="4GB")
    args = parser.parse_args()
    collect(args.output_dir, args.route, args.nproc, args.mem)


if __name__ == "__main__":
    main()
