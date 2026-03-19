"""
bond_features.csv → 보정 모델 학습

사용법:
    conda run -n delta_chem python scripts/train_model.py
"""
from delta_chem.ml.train import main

if __name__ == "__main__":
    main()
