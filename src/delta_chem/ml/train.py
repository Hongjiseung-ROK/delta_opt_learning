"""
bond_features.csv → GradientBoosting 보정 모델 학습

사용법:
    conda run -n delta_chem python scripts/train_model.py
    conda run -n delta_chem python scripts/train_model.py --features data/features/bond_features.csv
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


CAT_FEATURES = ["elem1", "elem2", "hybridization_1", "hybridization_2"]
NUM_FEATURES = ["bond_order", "is_in_ring", "ring_size", "mmff_length"]
TARGET = "dft_length"


def build_pipeline(n_estimators: int = 300, max_depth: int = 4) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def train(
    features_csv: str = "data/features/bond_features.csv",
    model_out: str = "models/bond_length_corrector.joblib",
    n_estimators: int = 300,
    max_depth: int = 4,
) -> None:
    df = pd.read_csv(features_csv)
    print(f"데이터: {len(df)}행  {df['mol_name'].nunique()}개 분자")
    print(f"결합 유형 분포:\n{df.groupby(['elem1','elem2','bond_order']).size().to_string()}\n")

    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df[TARGET]

    pipe = build_pipeline(n_estimators, max_depth)

    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"5-Fold CV MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f} Å")
    print(f"  (목표: < 0.005 Å for meaningful DFT correction)")

    # 전체 데이터로 최종 학습
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    # Feature importance
    model = pipe.named_steps["model"]
    feature_names = CAT_FEATURES + NUM_FEATURES
    importances = model.feature_importances_
    print("\nFeature importance:")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {fname:<20s} {imp:.4f}")

    # 저장
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    print(f"\n모델 저장: {model_out}")

    # ── 그래프 & CSV ──────────────────────────────────────────────────
    print("\n그래프 생성 중...")
    from delta_chem.viz import (
        plot_feature_importance, plot_parity, save_cv_scores_csv
    )
    plot_feature_importance(np.array(feature_names), importances)
    plot_parity(y.values, y_pred, mae_scores.mean())
    save_cv_scores_csv(mae_scores)


def main():
    parser = argparse.ArgumentParser(description="bond length correction 모델 학습")
    parser.add_argument("--features", default="data/features/bond_features.csv")
    parser.add_argument("--model-out", default="models/bond_length_corrector.joblib")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()
    train(args.features, args.model_out, args.n_estimators, args.max_depth)


if __name__ == "__main__":
    main()
