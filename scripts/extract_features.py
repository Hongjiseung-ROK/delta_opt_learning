"""
Gaussian .out 파일 → bond feature CSV + figures/01, 02

사용법:
    conda run -n delta_chem python scripts/extract_features.py
    conda run -n delta_chem python scripts/extract_features.py --results-dir data/raw --condition rdkit
"""
import argparse
from delta_chem.ml.feature_extractor import build_dataset
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from delta_chem.viz import (
    plot_mmff_vs_dft,
    plot_correction_distribution,
    save_correction_stats_csv,
)
from collect_data import MOLECULES as COLLECT_MOLECULES

# collect_data.py의 분자 목록에서 (name, smiles) 추출
MOLECULES = [(name, smiles) for name, smiles, *_ in COLLECT_MOLECULES]


def main():
    parser = argparse.ArgumentParser(description="Gaussian 결과 → bond features CSV + 그래프")
    parser.add_argument("--results-dir", default="data/raw")
    parser.add_argument("--condition", default="rdkit", choices=["rdkit", "xtb"])
    parser.add_argument("--output-csv", default="data/features/bond_features.csv")
    args = parser.parse_args()

    print(f"results_dir : {args.results_dir}")
    print(f"condition   : {args.condition}")

    df = build_dataset(
        results_dir=args.results_dir,
        molecule_list=MOLECULES,
        condition=args.condition,
        output_csv=args.output_csv,
    )

    if df.empty:
        print("데이터 없음 — 그래프 생성 건너뜀")
        return

    print(f"\n그래프 생성 중...")
    plot_mmff_vs_dft(df)
    plot_correction_distribution(df)
    save_correction_stats_csv(df)
    print("완료.")


if __name__ == "__main__":
    main()
