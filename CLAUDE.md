# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Undergraduate research: build a GradientBoosting ML model that corrects MMFF bond lengths toward B3LYP/6-31G(d) equilibrium values, reducing Gaussian DFT optimization steps.

**Full pipeline:**
```
SMILES → RDKit ETKDG+MMFF → ML bond length correction → Gaussian .com → Gaussian 16
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

## 서버 경로 및 Gaussian 16 실행 환경

- **서버**: CentOS 7, 20코어, RAM 62GB
- **Gaussian 16 실행파일**: `/opt/gaussian/g16/g16`
- **g16 환경 프로파일**: `/opt/gaussian/g16/bsd/g16.profile`
- xtb: 미사용 — `DELTA_XTB_EXE`로 오버라이드 가능
- 모든 경로 상수: `src/delta_chem/config.py`

### Gaussian 16 환경변수 설정

g16 실행 전 반드시 `GAUSS_EXEDIR`에 `bsd` 경로를 포함해야 한다 (없으면 실행 실패):

```bash
export g16root=/opt/gaussian
source /opt/gaussian/g16/bsd/g16.profile
# 결과: GAUSS_EXEDIR=/opt/gaussian/g16/bsd:/opt/gaussian/g16
```

또는 환경변수 직접 지정:
```bash
export GAUSS_EXEDIR=/opt/gaussian/g16/bsd:/opt/gaussian/g16
```

`config.py`는 이미 올바른 값으로 설정되어 있으므로 Python 스크립트에서는 별도 설정 불필요.

### .com 파일 리소스 설정 (20코어 노드 기준)

```
%NProcShared=20
%Mem=56GB
```

`gaussian_writer.py`의 `nproc`, `mem` 인자로 제어. 기본값은 각 스크립트에서 지정.

### Gaussian 16 on Linux 출력 파일

- Linux g16은 `<name>.log` 파일로 출력한다 (`stdout`은 비어 있음)
- `gaussian_runner.py`는 `.log` → `.out` 순으로 탐색하므로 별도 처리 불필요
- `data/raw/`에 기존재하는 `.out` 파일들은 **Windows Gaussian 09**로 생성된 것으로 포맷 동일

### Scratch 디렉토리

서버에 `/scratch` 없음. 기본 scratch는 `.com` 파일과 같은 디렉토리(cwd).
대용량 계산 시 `/tmp` 사용 권장:
```bash
export GAUSS_SCRDIR=/tmp
```

### SGE 클러스터 구성

- SGE 실행파일: `/opt/sge/bin/lx-amd64/` (PATH에 없으므로 직접 지정 필요)
- 큐 확인: `/opt/sge/bin/lx-amd64/qstat -f`

| 큐 | 노드 | 코어/노드 | 비고 |
|---|---|---|---|
| `20core.q` | node02~06 (5개) | 20코어 | 사용자 지정 노드 |
| `40core.q` | node07~20 (14개) | 40코어 | 타 사용자 점유 중 |

```bash
# 큐 상태 확인
/opt/sge/bin/lx-amd64/qstat -f

# 잡 제출 예시 (20core.q 지정)
/opt/sge/bin/lx-amd64/qsub -q 20core.q -pe smp 20 -cwd job.sh
```

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
│   ├── train_model.py          # bond_features.csv → models/*.joblib (기본: --target-mode delta)
│   ├── compare_target_modes.py # absolute vs delta 타겟 모드 비교 분석
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
- **출력**: `dft_length - mmff_length` (delta 모드, 기본값) 또는 `dft_length` (absolute 모드)
- **모델**: `sklearn.GradientBoostingRegressor` (n_estimators=300, max_depth=4, lr=0.05, subsample=0.8) + `OrdinalEncoder` Pipeline
- **모델 artifact**: `{"pipeline": pipe, "target_mode": "delta"|"absolute"}` 딕셔너리로 저장
- **5-Fold CV MAE**: delta 0.0022 ± 0.0004 Å / absolute 0.0026 ± 0.0009 Å (목표: < 0.005 Å)
- **학습 데이터**: 49개 분자, 529개 결합 (acetylene 제외)
- delta 모드가 CV MAE 16% 개선, `bond_order`·`elem2` 등 화학적 feature 기여 증가

## Gaussian 주의사항

- Gaussian 16 on Linux는 stdout 대신 `<name>.log` 파일에 출력한다
- subprocess 호출 시 반드시 `cwd=com_dir`로 설정하고 파일명만 넘겨야 경로 중복 버그가 없다
- `Elapsed time:` 줄이 실제 벽시계 시간 (Gaussian 16 on Linux에서 존재)
- `Job cpu time:` 은 코어×시간 합산값 — `log_parser.py`의 `cpu_seconds`는 이 값을 파싱함
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
3. scripts/train_model.py --exclude acetylene --target-mode delta  # 모델 학습 (delta 모드 권장)
4. scripts/benchmark_new_mols.py               # 훈련 미사용 분자로 검증
```
