#!/bin/bash
#$ -q 20core.q
#$ -pe smp 20
#$ -N freesolv_node1
#$ -cwd
#$ -j y
#$ -o /home/hjs0603/delta_opt_learning/logs/freesolv_node1_$JOB_ID.log

# FreeSolv 분산 계산 — 노드 1: idx 0~320 (321개)
# 제출: /opt/sge/bin/lx-amd64/qsub scripts/sge_freesolv_node1.sh

echo "=== Job Start: $(date) ==="
echo "Host: $(hostname)"
echo "Cores: $(nproc)"

# Gaussian 16 환경 설정
export g16root=/opt/gaussian
source /opt/gaussian/g16/bsd/g16.profile
export GAUSS_SCRDIR=/tmp

# conda 경로 설정 (컴퓨트 노드에는 PATH 미등록)
export PATH=/home/hjs0603/miniconda3/bin:$PATH

# 프로젝트 루트로 이동
cd /home/hjs0603/delta_opt_learning

# Gaussian 계산만 수행 (feature 추출은 양쪽 완료 후 별도 실행)
conda run -n delta_chem python scripts/collect_freesolv.py \
    --start-idx 0 \
    --end-idx 321 \
    --nproc 20 \
    --mem 56GB \
    --skip-extract

echo "=== Job End: $(date) ==="
