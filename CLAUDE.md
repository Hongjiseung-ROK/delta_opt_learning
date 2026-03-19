# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Undergraduate research: build a GradientBoosting ML model that corrects MMFF bond lengths toward B3LYP/6-31G(d) equilibrium values, reducing Gaussian DFT optimization steps.

**Full pipeline:**
```
SMILES → RDKit ETKDG+MMFF → ML bond length correction → Gaussian .com → Gaussian 09
```

## Environment Setup

```bash
# 최초 환경 생성
conda env create -f environment.yml

# 패키지 editable 설치 (최초 1회)
conda run -n delta_chem pip install -e .

# 스크립트 실행 방법 (bash에서 conda activate가 안 되므로 conda run 사용)
conda run -n delta_chem python scripts/pipeline.py "CCO" --name ethanol
conda run -n delta_chem python scripts/benchmark.py
conda run -n delta_chem python scripts/extract_features.py
conda run -n delta_chem python scripts/train_model.py
```

## Local Paths (Windows)

- Gaussian 09: `C:\G09W\g09.exe` — 환경변수 `DELTA_G09_EXE`로 오버라이드
- xtb: `C:\Users\1172\miniconda3\envs\delta_chem\Library\bin\xtb.exe` — `DELTA_XTB_EXE`로 오버라이드
- 모든 경로 상수: `src/delta_chem/config.py`

## Repository Structure

```
delta_chem/
├── src/delta_chem/
│   ├── config.py               # G09_EXE, XTB_EXE 경로 상수
│   ├── chem/                   # 화학 I/O 계층
│   │   ├── smiles_to_xyz.py    # SMILES → RDKit ETKDG+MMFF → XYZ
│   │   ├── xtb_optimizer.py    # XYZ → xtb GFN2 opt → XYZ
│   │   ├── gaussian_writer.py  # XYZ → Gaussian .com
│   │   ├── gaussian_runner.py  # g09 subprocess 실행 → .out 경로 반환
│   │   └── log_parser.py       # .out 파싱: 스텝수/시간/최종좌표
│   └── ml/                     # ML 계층
│       ├── feature_extractor.py # MMFF+DFT 좌표 → bond feature DataFrame
│       ├── train.py             # GradientBoosting 학습
│       └── corrector.py         # 모델 적용: MMFF mol → 보정된 mol
├── scripts/
│   ├── pipeline.py             # CLI: SMILES → .com (--ml-correct 플래그)
│   ├── benchmark.py            # 조건별 Gaussian 비교 (rdkit/xtb/ml)
│   ├── extract_features.py     # data/raw/ → data/features/bond_features.csv
│   ├── train_model.py          # bond_features.csv → models/*.joblib
│   └── test_pipeline.py        # 스모크 테스트
├── data/
│   ├── raw/                    # Gaussian .out 파일 (gitignored)
│   └── features/               # bond_features.csv (gitignored)
├── models/                     # 학습된 .joblib 모델 (gitignored)
├── notebooks/                  # 분석용 Jupyter 노트북
├── pyproject.toml
└── environment.yml
```

## ML Model Architecture

- **입력 feature** (결합 1개당): `elem1`, `elem2`, `bond_order`, `hybridization_1`, `hybridization_2`, `is_in_ring`, `ring_size`, `mmff_length`
- **출력**: `dft_length` (B3LYP/6-31G(d) 평형 결합 길이, Å)
- **모델**: `sklearn.GradientBoostingRegressor` + `OrdinalEncoder` Pipeline
- **목표 MAE**: < 0.005 Å

## Gaussian 09 주의사항

- Gaussian 09 on Windows는 stdout 대신 `<name>.out` 파일에 직접 쓴다
- subprocess 호출 시 반드시 `cwd=com_dir`로 설정하고 파일명만 넘겨야 경로 중복 버그가 없다
- `Job cpu time:` 을 벽시계 시간 대용으로 사용 (Elapsed time 없음)
- 최적화 스텝: `"Step number   N out of"` 패턴의 마지막 N
- 최종 DFT 좌표: `"Standard orientation:"` 블록의 마지막 출현

## ML 학습 데이터 수집 순서

```
1. scripts/benchmark.py --conditions rdkit   # Gaussian rdkit 조건 실행
2. scripts/extract_features.py               # .out → bond_features.csv
3. scripts/train_model.py                    # 모델 학습
4. scripts/benchmark.py --conditions rdkit ml  # ML 보정 효과 검증
```
