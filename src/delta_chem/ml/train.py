"""
bond_features.csv → GradientBoosting 보정 모델 학습

사용법:
    conda run -n delta_chem python scripts/train_model.py
    conda run -n delta_chem python scripts/train_model.py --target-mode delta
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
NUM_FEATURES = ["bond_order", "is_in_ring", "ring_size",
                "mmff_length", "mmff_angle_1", "mmff_angle_2"]


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
    exclude: list | None = None,
    target_mode: str = "absolute",   # "absolute" | "delta"
) -> dict:
    """
    모델을 학습하고 결과 dict를 반환한다.

    target_mode:
      "absolute" — dft_length를 직접 예측 (기존 방식)
      "delta"    — (dft_length - mmff_length) 보정량을 예측
    """
    df = pd.read_csv(features_csv)
    if exclude:
        df = df[~df["mol_name"].isin(exclude)]
        print(f"제외 분자: {exclude}")
    print(f"데이터: {len(df)}행  {df['mol_name'].nunique()}개 분자")
    print(f"타겟 모드: {target_mode}")

    X = df[CAT_FEATURES + NUM_FEATURES]

    if target_mode == "delta":
        y = df["dft_length"] - df["mmff_length"]
        print(f"타겟(delta) 통계: mean={y.mean():.5f} std={y.std():.5f} Å")
    else:
        y = df["dft_length"]
        print(f"타겟(dft_length) 통계: mean={y.mean():.5f} std={y.std():.5f} Å")

    pipe = build_pipeline(n_estimators, max_depth)

    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_raw = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")

    # delta 모드 CV MAE는 dft_length 기준으로 환산 (동일 스케일 비교)
    # delta와 absolute 모두 실제 dft_length 오차로 해석 가능 (mmff_length가 feature이므로)
    print(f"5-Fold CV MAE ({target_mode}): {cv_raw.mean():.4f} ± {cv_raw.std():.4f} Å")

    # 전체 데이터로 최종 학습
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    # delta 모드: train MAE를 dft_length 공간에서도 확인
    if target_mode == "delta":
        y_dft_pred = df["mmff_length"] + y_pred
        train_mae_dft = float(np.abs(y_dft_pred - df["dft_length"]).mean())
        print(f"Train MAE (dft_length 공간): {train_mae_dft:.4f} Å")

    # Feature importance
    model = pipe.named_steps["model"]
    feature_names = CAT_FEATURES + NUM_FEATURES
    importances = model.feature_importances_
    print("\nFeature importance:")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {fname:<22s} {imp:.4f}  {bar}")

    # 저장: pipeline + target_mode를 함께 묶어 저장
    artifact = {"pipeline": pipe, "target_mode": target_mode}
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)
    print(f"\n모델 저장: {model_out}")

    return {
        "pipe": pipe,
        "target_mode": target_mode,
        "cv_mae": cv_raw,
        "feature_names": np.array(feature_names),
        "importances": importances,
        "y_true": y.values,
        "y_pred": y_pred,
        "df": df,
    }


def main():
    parser = argparse.ArgumentParser(description="bond length correction 모델 학습")
    parser.add_argument("--features", default="data/features/bond_features.csv")
    parser.add_argument("--model-out", default="models/bond_length_corrector.joblib")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--exclude", nargs="*", default=None)
    parser.add_argument("--target-mode", choices=["absolute", "delta"], default="absolute",
                        help="absolute: dft_length 직접 예측 / delta: (dft-mmff) 보정량 예측")
    args = parser.parse_args()

    result = train(
        args.features, args.model_out,
        args.n_estimators, args.max_depth,
        args.exclude, args.target_mode,
    )

    from delta_chem.viz import plot_feature_importance, plot_parity, save_cv_scores_csv
    plot_feature_importance(result["feature_names"], result["importances"])
    plot_parity(result["y_true"], result["y_pred"], result["cv_mae"].mean())
    save_cv_scores_csv(result["cv_mae"])


if __name__ == "__main__":
    main()
