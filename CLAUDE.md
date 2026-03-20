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
conda run -n delta_chem python scripts/collect_data.py
conda run -n delta_chem python scripts/extract_features.py
conda run -n delta_chem python scripts/train_model.py --exclude acetylene
conda run -n delta_chem python scripts/benchmark_new_mols.py
conda run -n delta_chem python scripts/benchmark_acetylene.py
```

## Local Paths (Windows)

- Gaussian 09: `C:\G09W\g09.exe` — 환경변수 `DELTA_G09_EXE`로 오버라이드
- xtb: `C:\Users\1172\miniconda3\envs\delta_chem\Library\bin\xtb.exe` — `DELTA_XTB_EXE`로 오버라이드 (현재 미사용)
- 모든 경로 상수: `src/delta_chem/config.py`

## Repository Structure

```
delta_chem/
├── src/delta_chem/
│   ├── config.py               # G09_EXE, XTB_EXE 경로 상수
│   ├── chem/                   # 화학 I/O 계층
│   │   ├── smiles_to_xyz.py    # SMILES → MMFF XYZ + mol_to_xyz() 공통 유틸리티
│   │   ├── xtb_optimizer.py    # xtb GFN2 최적화 (현재 미사용, 레거시 보존)
│   │   ├── gaussian_writer.py  # XYZ → Gaussian .com
│   │   ├── gaussian_runner.py  # g09 subprocess 실행 → .out 경로 반환
│   │   └── log_parser.py       # .out 파싱: 스텝수/시간/좌표
│   └── ml/                     # ML 계층
│       ├── feature_extractor.py # MMFF+DFT 좌표 → bond feature DataFrame
│       ├── train.py             # GradientBoosting 학습
│       └── corrector.py         # 모델 적용 (배치 예측, 모듈 레벨 캐싱)
├── scripts/
│   ├── pipeline.py             # CLI: SMILES → .com (--ml-correct 플래그)
│   ├── collect_data.py         # 50개 분자 Gaussian 계산 (rdkit 조건)
│   ├── extract_features.py     # data/raw/ → data/features/bond_features.csv
│   ├── train_model.py          # bond_features.csv → models/*.joblib
│   ├── benchmark.py            # 조건별 Gaussian 비교 (rdkit/ml)
│   ├── benchmark_new_mols.py   # 훈련 미사용 5개 분자 벤치마크
│   └── benchmark_acetylene.py  # Acetylene 케이스 스터디 (MMFF vs ML)
├── notebooks/
│   └── pipeline_comparison.ipynb  # 대화형 SMILES 입력 → 파이프라인 비교
├── data/
│   ├── raw/                    # Gaussian .out 파일 (gitignored)
│   └── features/               # bond_features.csv (gitignored)
├── models/                     # 학습된 .joblib 모델 (gitignored)
├── figures/                    # 생성된 그래프 및 CSV (01~07)
└── environment.yml
```

## ML Model Architecture

- **입력 feature** (결합 1개당): `elem1`, `elem2`, `bond_order`, `hybridization_1`, `hybridization_2`, `is_in_ring`, `ring_size`, `mmff_length`
- **출력**: `dft_length` (B3LYP/6-31G(d) 평형 결합 길이, Å)
- **모델**: `sklearn.GradientBoostingRegressor` (n_estimators=300, max_depth=4, lr=0.05, subsample=0.8) + `OrdinalEncoder` Pipeline
- **5-Fold CV MAE**: 0.0026 ± 0.0009 Å (목표: < 0.005 Å)
- **학습 데이터**: 49개 분자, 529개 결합 (acetylene 제외)

## Gaussian 09 주의사항

- Gaussian 09 on Windows는 stdout 대신 `<name>.out` 파일에 직접 쓴다
- subprocess 호출 시 반드시 `cwd=com_dir`로 설정하고 파일명만 넘겨야 경로 중복 버그가 없다
- `Job cpu time:` 을 벽시계 시간 대용으로 사용 (Elapsed time 없음)
- 최적화 스텝: `"Step number N out of"` 패턴의 **max 값** 사용 (last 값 아님)
  → `opt freq` 계산 시 freq 이후 추가 Berny 루프가 "Step number 1 out of 2"를 생성하므로,
     max()를 써야 실제 최적화 스텝 수를 얻는다
- 최종 DFT 좌표: `"Standard orientation:"` 블록의 **마지막** 출현
- MMFF 초기 좌표: `"Input orientation:"` 블록의 **첫 번째** 출현 (feature 추출용)

## Feature Extraction 주의사항

- MMFF 좌표는 ETKDGv3를 재실행하지 않고 Gaussian .out의 `Input orientation` 블록에서 직접 파싱한다
- ETKDGv3는 stochastic하므로 재실행 시 Gaussian에 실제 제출된 구조와 다른 conformer가 생성될 수 있다
- 혼성화(hybridization) 정보 파악을 위해서만 ETKDGv3+MMFF를 임시 실행한다 (좌표는 사용 안 함)

## 공통 유틸리티

- `mol_to_xyz(mol, path, title)`: `src/delta_chem/chem/smiles_to_xyz.py` — RDKit mol → XYZ 파일
  여러 스크립트에서 공유, 각 스크립트에서 별도 정의 금지

## ML 학습 데이터 수집 순서

```
1. scripts/collect_data.py                      # Gaussian rdkit 조건 실행
2. scripts/extract_features.py                  # .out → bond_features.csv
3. scripts/train_model.py --exclude acetylene   # 모델 학습
4. scripts/benchmark_new_mols.py               # 훈련 미사용 분자로 검증
```
