#!/bin/bash
#$ -q 20core.q
#$ -l hostname=node02
#$ -pe smp 4
#$ -N post_freesolv
#$ -j y
#$ -o /home/hjs0603/delta_opt_learning/logs/post_freesolv.log
#$ -hold_jid 79663,79664

export PATH=/home/hjs0603/miniconda3/bin:$PATH
export g16root=/opt/gaussian
source /opt/gaussian/g16/bsd/g16.profile
export GAUSS_SCRDIR=/tmp
cd /home/hjs0603/delta_opt_learning

echo "========================================"
echo "  Post-FreeSolv Pipeline"
echo "  Start: $(date)"
echo "  Host:  $(hostname)"
echo "========================================"

# ── 1. FreeSolv feature 추출 (전체 분자) ─────────────────────────────────
echo ""
echo "[1/4] FreeSolv feature 추출..."
conda run -n delta_chem python scripts/collect_freesolv.py \
    --extract-only \
    --output-dir data/raw \
    --output-csv data/features/bond_features_freesolv_full.csv

if [ $? -ne 0 ]; then
    echo "ERROR: feature 추출 실패"
    exit 1
fi

# ── 2. 원래 50개 분자 + FreeSolv 전체 합산 CSV 생성 ──────────────────────
echo ""
echo "[2/4] 전체 feature CSV 병합..."
conda run -n delta_chem python - <<'PYEOF'
import pandas as pd
from pathlib import Path

base    = Path("data/features/bond_features.csv")
freesolv = Path("data/features/bond_features_freesolv_full.csv")
out     = Path("data/features/bond_features_full.csv")

dfs = []
if base.exists():
    df_base = pd.read_csv(base)
    dfs.append(df_base)
    print(f"  기존 데이터: {len(df_base)}행 ({df_base['mol_name'].nunique()}개 분자)")

if freesolv.exists():
    df_fs = pd.read_csv(freesolv)
    dfs.append(df_fs)
    print(f"  FreeSolv:   {len(df_fs)}행 ({df_fs['mol_name'].nunique()}개 분자)")

if not dfs:
    print("ERROR: CSV 없음")
    exit(1)

combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["mol_name","bond_idx"])
out.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out, index=False)
print(f"\n병합 완료: {out}")
print(f"  총 {len(combined)}행 / {combined['mol_name'].nunique()}개 분자")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: CSV 병합 실패"
    exit 1
fi

# ── 3a. GBM 재학습 ────────────────────────────────────────────────────────
echo ""
echo "[3/4] GBM 재학습 (전체 데이터)..."
conda run -n delta_chem python scripts/train_model.py \
    --features   data/features/bond_features_full.csv \
    --model-out  models/bond_length_corrector_delta_full.joblib \
    --target-mode delta \
    --exclude acetylene

if [ $? -ne 0 ]; then
    echo "ERROR: GBM 학습 실패"
    exit 1
fi

# ── 3b. GNN 재학습 ────────────────────────────────────────────────────────
echo ""
echo "[4/4] GNN 재학습 (전체 데이터)..."
conda run -n delta_chem python scripts/gnn/train_gnn.py \
    --csv        data/features/bond_features_full.csv \
    --model-out  models/bond_length_corrector_gnn_full.joblib \
    --epochs     300 \
    --lr         1e-3 \
    --batch-size 32 \
    --hidden-dim 128 \
    --n-layers   4 \
    --exclude    acetylene \
    --device     cpu

if [ $? -ne 0 ]; then
    echo "ERROR: GNN 학습 실패"
    exit 1
fi

# ── 4. 벤치마크 ───────────────────────────────────────────────────────────
echo ""
echo "[5/5] 벤치마크 실행..."

# benchmark_new_mols.py의 모델 경로를 full 모델로 교체하여 실행
conda run -n delta_chem python - <<'PYEOF'
import sys, os
sys.path.insert(0, ".")

# benchmark_new_mols 모듈을 직접 import해서 모델 경로만 바꿔 실행
import importlib.util, types

# 모델 경로 패치
import scripts.benchmark_new_mols as bm

# 원래 PIPELINES를 full 모델로 교체
bm.MODEL_OLD = "models/bond_length_corrector_delta.joblib"          # GBM-50 (비교 기준)
bm.MODEL_EXP = "models/bond_length_corrector_delta_full.joblib"     # GBM-full
bm.MODEL_GNN = "models/bond_length_corrector_gnn_full.joblib"       # GNN-full

bm.PIPELINES = [
    ("rdkit",     None,          None),
    ("gbm_50",    bm.MODEL_OLD,  "gbm"),
    ("gbm_full",  bm.MODEL_EXP,  "gbm"),
    ("gnn_full",  bm.MODEL_GNN,  "gnn"),
]

# 결과를 별도 디렉토리에 저장
from pathlib import Path
bm.OUTDIR = Path("data/raw/benchmark_full")
bm.FIGDIR = Path("figures")

import argparse
args = argparse.Namespace(nproc=20, mem="56GB", rerun=False)

df = bm.run_benchmark(args.nproc, args.mem, args.rerun)
bm.report_and_plot(df)

# CSV 경로 변경 (report_and_plot 내부 경로 override)
df.to_csv("figures/10_benchmark_full.csv", index=False)
print("\n결과 저장: figures/10_benchmark_full.csv")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: 벤치마크 실패"
    exit 1
fi

# ── 5. Git commit & push → branch sub ────────────────────────────────────
echo ""
echo "[Git] 브랜치 sub에 결과 커밋 및 push..."

cd /home/hjs0603/delta_opt_learning

# sub 브랜치 생성 또는 체크아웃 (이미 있으면 체크아웃)
git checkout sub 2>/dev/null || git checkout -b sub

# master 최신 내용 병합 (충돌 없이)
git merge master --no-edit

# 스테이징: data/raw 제외, 나머지 전부
git add \
    README.md \
    CLAUDE.md \
    crit.md \
    .gitignore \
    environment.yml \
    src/ \
    scripts/ \
    notebooks/ \
    figures/

# 변경 없으면 커밋 생략
if git diff --cached --quiet; then
    echo "변경 사항 없음 — 커밋 생략"
else
    git commit -m "$(cat <<'GITMSG'
feat: FreeSolv full dataset retraining + GNN pipeline

- FreeSolv 642분자 Gaussian 계산 완료 후 GBM/GNN 재학습
- GNN (EC-MPNN, hidden=128, layers=4) 추가
- 4-way 벤치마크: MMFF / GBM-50 / GBM-full / GNN-full
- bond angle feature (mmff_angle_1/2) 추가
- scripts/gnn/ 디렉토리: dataset, model, train, corrector
- crit.md: 현황 분석 및 개선 로드맵
- README 전면 업데이트

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
GITMSG
)"
fi

# push
git push -u origin sub

echo ""
echo "========================================"
echo "  All Done: $(date)"
echo "  Branch: sub → origin/sub"
echo "========================================"
