#!/bin/bash
#$ -q 20core.q
#$ -pe smp 4
#$ -N gnn_train
#$ -j y
#$ -o /home/hjs0603/delta_opt_learning/logs/gnn_train.log
#$ -cwd

export PATH=/home/hjs0603/miniconda3/bin:$PATH
cd /home/hjs0603/delta_opt_learning

echo "===== GNN Training Start ====="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(conda run -n delta_chem python --version 2>&1)"
echo ""

# torch / torch_geometric 버전 확인
conda run -n delta_chem python -c "
import torch
print('torch:', torch.__version__)
import torch_geometric
print('torch_geometric:', torch_geometric.__version__)
"

echo ""
echo "===== 5-Fold CV Training ====="
conda run -n delta_chem python scripts/gnn/train_gnn.py \
    --csv    data/features/bond_features_combined.csv \
    --model-out models/bond_length_corrector_gnn.joblib \
    --epochs 200 \
    --lr     1e-3 \
    --batch-size 16 \
    --hidden-dim 64 \
    --n-layers 3 \
    --exclude acetylene \
    --device cpu

echo ""
echo "===== Done: $(date) ====="
