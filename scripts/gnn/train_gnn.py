"""
GNN 모델 학습 스크립트

5-Fold CV로 GBM과 동일 조건에서 성능 비교.

사용법:
    conda run -n delta_chem python scripts/gnn/train_gnn.py
    conda run -n delta_chem python scripts/gnn/train_gnn.py --epochs 200 --lr 5e-4
    conda run -n delta_chem python scripts/gnn/train_gnn.py --csv data/features/bond_features_combined.csv
"""
import argparse
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

from scripts.gnn.dataset import build_dataset
from scripts.gnn.model import BondLengthGNN, count_parameters


def edge_mae(pred: torch.Tensor, y: torch.Tensor) -> float:
    """엣지(결합) 레벨 MAE 계산. 양방향 엣지를 평균."""
    # 양방향 엣지 → 쌍수 짝수개 → 매 2개씩 평균
    return float(torch.abs(pred - y).mean().item())


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_bonds = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = nn.functional.mse_loss(pred, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss   += loss.item() * batch.y.size(0)
        total_bonds  += batch.y.size(0)
    return total_loss / total_bonds


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_pred, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        all_pred.append(pred.cpu())
        all_y.append(batch.y.cpu())
    all_pred = torch.cat(all_pred)
    all_y    = torch.cat(all_y)
    mae = float(torch.abs(all_pred - all_y).mean())
    return mae


def run_cv(
    graphs,
    n_splits: int = 5,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 16,
    hidden_dim: int = 64,
    n_layers: int = 3,
    device_str: str = "cpu",
    verbose: bool = True,
) -> tuple[list[float], object]:
    """
    5-Fold CV 실행. 마지막 fold 모델 반환.
    Returns: (fold MAE 리스트, 마지막 fold의 model)
    """
    device = torch.device(device_str)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = list(range(len(graphs)))

    fold_maes = []
    last_model = None

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs   = [graphs[i] for i in val_idx]

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False, num_workers=0)

        model = BondLengthGNN(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-5)

        best_val_mae = float("inf")
        best_state   = None

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_mae    = eval_epoch(model, val_loader, device)
            scheduler.step(val_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

            if verbose and epoch % 30 == 0:
                lr_cur = optimizer.param_groups[0]["lr"]
                print(f"  Fold {fold_idx+1} | Epoch {epoch:>3d} | "
                      f"TrainLoss {train_loss:.6f} | ValMAE {val_mae:.4f} Å | lr={lr_cur:.2e}")

        # 최적 가중치 복원
        model.load_state_dict(best_state)
        fold_maes.append(best_val_mae)
        last_model = model
        if verbose:
            print(f"  Fold {fold_idx+1} 최적 Val MAE: {best_val_mae:.4f} Å")

    return fold_maes, last_model


def train_full(
    graphs,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 16,
    hidden_dim: int = 64,
    n_layers: int = 3,
    device_str: str = "cpu",
    verbose: bool = True,
) -> object:
    """전체 데이터로 최종 모델 학습"""
    device = torch.device(device_str)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True, num_workers=0)

    model = BondLengthGNN(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-5)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device)
        scheduler.step(train_loss)
        if verbose and epoch % 30 == 0:
            lr_cur = optimizer.param_groups[0]["lr"]
            print(f"  Full Train | Epoch {epoch:>3d} | Loss {train_loss:.6f} | lr={lr_cur:.2e}")

    return model


def save_model(model, model_path: str, cv_mae: list[float]):
    """GNN 모델 저장 (GBM artifact와 호환되는 dict 형식)"""
    import joblib
    artifact = {
        "type":       "gnn",
        "model_state": model.state_dict(),
        "model_kwargs": {
            "hidden_dim": model.convs[0].msg_mlp[0].out_features,
            "n_layers":   len(model.convs),
        },
        "target_mode": "delta",
        "cv_mae":      np.array(cv_mae),
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"\n모델 저장: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="GNN bond length corrector 학습")
    parser.add_argument("--csv",       default="data/features/bond_features_combined.csv")
    parser.add_argument("--model-out", default="models/bond_length_corrector_gnn.joblib")
    parser.add_argument("--epochs",    type=int,   default=150)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch-size",type=int,   default=16)
    parser.add_argument("--hidden-dim",type=int,   default=64)
    parser.add_argument("--n-layers",  type=int,   default=3)
    parser.add_argument("--exclude",   nargs="*",  default=["acetylene"])
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"device={args.device}, epochs={args.epochs}, lr={args.lr}, "
          f"hidden={args.hidden_dim}, layers={args.n_layers}")

    # 데이터 로드
    t0 = time.time()
    graphs = build_dataset(args.csv, exclude=args.exclude)
    if not graphs:
        print("데이터 없음. 종료.")
        return
    total_bonds = sum(g.y.size(0) // 2 for g in graphs)  # 양방향이므로 /2
    print(f"총 {len(graphs)}개 분자, {total_bonds}개 결합")

    model = BondLengthGNN(hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    print(f"GNN 파라미터: {count_parameters(model):,}개")

    # 5-Fold CV
    print("\n===== 5-Fold Cross-Validation =====")
    cv_maes, _ = run_cv(
        graphs,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        device_str=args.device,
    )
    cv_arr = np.array(cv_maes)
    print(f"\n5-Fold CV MAE: {cv_arr.mean():.4f} ± {cv_arr.std():.4f} Å")
    print(f"  각 fold: {[f'{v:.4f}' for v in cv_maes]}")

    # 전체 데이터로 최종 학습
    print("\n===== 최종 모델 학습 (전체 데이터) =====")
    final_model = train_full(
        graphs,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        device_str=args.device,
    )

    save_model(final_model, args.model_out, cv_maes)
    print(f"\n소요 시간: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
