# delta_opt_learning

**Accelerating Gaussian DFT Geometry Optimization via MMFF → B3LYP/6-31G(d) Bond Length Correction**

Undergraduate Research Project | B3LYP/6-31G(d) | Python · RDKit · Gaussian 09 · scikit-learn

---

## Overview

The convergence speed of DFT (Density Functional Theory) geometry optimization in Gaussian 09 varies significantly depending on the quality of the initial structure. The RDKit ETKDG+MMFF initial structure, which is commonly used for its convenience, exhibits a systematic discrepancy between the MMFF force-field equilibrium bond lengths and those at the B3LYP/6-31G(d) level.

This project aims to learn this discrepancy using a **Gradient Boosting machine learning model**, and by correcting the initial structure before running Gaussian, reduce the number of optimization steps and computation time.

---

## Pipeline

```
SMILES
  │
  ▼
RDKit ETKDG + MMFF optimization        ← initial 3D structure generation
  │
  ▼
ML Bond Length Correction              ← per-bond-type MMFF→DFT correction
  │
  ▼
Gaussian 09 B3LYP/6-31G(d) opt        ← actual DFT optimization
```

### Quick Start

```bash
# Create environment
conda env create -f environment.yml
conda run -n delta_chem pip install -e .

# SMILES → Gaussian .com (with ML correction)
conda run -n delta_chem python scripts/pipeline.py "CCO" --name ethanol --ml-correct

# Full data collection (requires Gaussian)
conda run -n delta_chem python scripts/collect_data.py

# Feature extraction → model training
conda run -n delta_chem python scripts/extract_features.py
conda run -n delta_chem python scripts/train_model.py --exclude acetylene

# Benchmark
conda run -n delta_chem python scripts/benchmark_new_mols.py
```

---

## Methodology

### 1. Training Data Generation

Two geometries were collected for 49 simple organic molecules (alkanes, alkenes, alkynes, aromatics, heterocycles, alcohols, ethers, ketones, carboxylic acids, esters, amines, halogens, etc.).

- **MMFF structure**: the `Input orientation` block from Gaussian `.out` files (parsed directly from the actual submitted MMFF structure)
- **DFT structure**: the `Standard orientation` block after Gaussian 09 B3LYP/6-31G(d) optimization

> **Note:** Obtaining MMFF coordinates by parsing `Input orientation` rather than re-running ETKDGv3 ensures consistency of the training data.

### 2. Feature Engineering

8 features were extracted per bond:

| Feature | Description |
|---------|-------------|
| `elem1`, `elem2` | The two elements forming the bond (alphabetically sorted) |
| `bond_order` | Bond order (1.0 / 1.5 / 2.0 / 3.0) |
| `hybridization_1/2` | Hybridization of each atom (SP / SP2 / SP3) |
| `is_in_ring` | Whether the bond is part of a ring |
| `ring_size` | Smallest ring size |
| `mmff_length` | MMFF bond length (Å) |

**Prediction target**: B3LYP/6-31G(d) bond length (Å)

### 3. Model

A sklearn `Pipeline` combining `scikit-learn GradientBoostingRegressor` with `OrdinalEncoder` for categorical features was used.

```
n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8
```

---

## Results

### Dataset

| Item | Value |
|------|-------|
| Training molecules | **49** |
| Training bonds | **529** |
| Bond types | 15 |
| Excluded molecule | acetylene (initial structure failed during collection) |

### MMFF → DFT Correction by Bond Type

| Bond type | n | Mean correction (Å) | Std dev (Å) |
|-----------|---|---------------------|-------------|
| C-Cl(s) | 1 | **+0.03654** | — |
| C-S(ar) | 2 | **+0.02452** | 0.000 |
| C-C(s) | 77 | **+0.00973** | 0.00649 |
| C-C(t) | 2 | +0.00556 | 0.00127 |
| C-H(s) | 323 | +0.00214 | 0.00285 |
| C-C(ar) | 48 | +0.00252 | 0.01102 |
| C-O(s) | 26 | −0.00036 | 0.00804 |
| **C=O(d)** | 11 | **−0.01197** | 0.00407 |
| H-O(s) | 10 | −0.00285 | 0.00120 |

It was confirmed that MMFF consistently underestimates C-C single bonds, while overestimating C=O double bonds.

### MMFF vs DFT Scatter Plot

![MMFF vs DFT](figures/01_mmff_vs_dft.png)

### Correction Distribution

![Correction Distribution](figures/02_correction_dist.png)

### Target Mode Comparison: Absolute vs Delta

The prediction target was changed from `dft_length` (absolute) to `dft_length - mmff_length` (delta) and compared.

![Target Mode Comparison](figures/05_target_mode_comparison.png)

| Target mode | CV MAE (Å) | CV std (Å) | mmff_length importance |
|-------------|-----------|-----------|------------------------|
| absolute | 0.00261 | 0.00088 | 98.3% |
| **delta** | **0.00219** | **0.00043** | **81.3%** |
| Improvement | **+16%** | **+51%** | — |

In delta mode:
- CV MAE improved by 16%, stability (std) improved by 51%
- Increased contribution from chemical features such as `bond_order` (7.3%), `elem2` (5.5%), `ring_size` (2.7%)
- The model begins to learn bond-type-specific correction patterns beyond linear scaling

→ **The current default model uses the delta mode.**

### Model Performance (5-Fold Cross-Validation, delta mode)

| Fold | MAE (Å) |
|------|---------|
| 1 | 0.00217 |
| 2 | 0.00251 |
| 3 | 0.00183 |
| 4 | 0.00221 |
| 5 | 0.00224 |
| **Mean** | **0.0022 ± 0.0004** |

Achieved B3LYP/6-31G(d) bond length prediction MAE of **0.0022 Å**. Target (< 0.005 Å) achieved.

### Feature Importance

![Feature Importance](figures/03_feature_importance.png)

In delta mode, `mmff_length` importance decreased to 81.3%, and the contribution of chemical features such as `bond_order` (7.3%) and `elem2` (5.5%) increased.

### Parity Plot (Predicted vs Actual)

![Parity Plot](figures/04_parity.png)

### Acetylene Case Study

![Acetylene Correction](figures/06_acetylene_correction.png)

| | MMFF | ML corrected | DFT reference |
|---|---|---|---|
| C≡C bond length | 1.2003 Å | 1.2056 Å | 1.2050 Å |
| Error | 0.0047 Å | **0.0006 Å** | — |
| Error reduction | — | **86%** | — |
| Gaussian steps | 3 | 3 | — |

### Benchmark on 5 Unseen Molecules

![New Molecules Benchmark](figures/07_benchmark_new_mols.png)

Re-evaluated with the delta mode model (compared against previous absolute model results):

| Molecule | Functional group | MMFF steps | ML steps | Step change | CPU change |
|----------|-----------------|-----------|---------|-------------|------------|
| cyclopentane | 5-membered ring | 5 | 5 | 0 *(previously: −1 degraded)* | **+7%** |
| 1-butanol | long-chain alcohol | 7 | 7 | 0 | 0% |
| acetophenone | aromatic ketone | 4 | 4 | 0 | +1% |
| acrolein | conjugated carbonyl | 5 | **4** | **+20%** | **+19%** |
| dimethylsulfoxide | S=O bond | 6 | 6 | 0 | **+8%** |

In delta mode, the step degradation for cyclopentane was resolved. This is interpreted as a result of more conservative correction values leading to reduced geometric distortion in the 5-membered ring.

---

## Limitations and Future Work

### Current Limitations

1. **Over-reliance on a single feature**: After introducing delta mode, `mmff_length` importance decreased from 98.3% to 81.3%, but it remains dominant. Thousands of bond data points would be needed to sufficiently learn non-linear electronic effects (resonance, electronegativity).

2. **Simple features**: Current features are local and do not reflect the electronic structure of the entire molecule. Global features such as neighboring atom environments, partial charges, and resonance structures need to be added.

3. **Limitations of coordinate correction approach**: The DFS tree traversal-based bond scaling treats each bond independently, so bond angles and dihedral angles are not corrected.

4. **Dependence on training distribution**: Bond types that are absent or sparse in the training data, such as 5-membered rings (cyclopentane) and S=O bonds (DMSO), show unstable correction quality.

5. **Element limitations**: The current training set covers C, H, O, N, S, Cl. Metal complexes and elements such as P/F/Br are not supported.

### Future Work

- [ ] GNN (Graph Neural Network)-based model — using the full molecular structure as input
- [ ] Add bond angle correction
- [ ] Expand training data with more molecules covering diverse bond environments
- [ ] Validate edge cases such as strained rings and hydrogen bonding

---

## Repository Structure

```
delta_opt_learning/
├── src/delta_chem/
│   ├── config.py               # path constants (G09_EXE, etc.)
│   ├── chem/
│   │   ├── smiles_to_xyz.py    # SMILES → MMFF XYZ, mol_to_xyz utility
│   │   ├── gaussian_writer.py  # XYZ → Gaussian .com
│   │   ├── gaussian_runner.py  # Gaussian 09 subprocess execution
│   │   └── log_parser.py       # .out parsing (steps/time/geometry)
│   ├── ml/
│   │   ├── feature_extractor.py # MMFF+DFT coordinates → bond feature DataFrame
│   │   ├── train.py             # GradientBoosting training
│   │   └── corrector.py         # ML correction application (batch prediction, model caching)
│   └── viz.py                  # matplotlib visualization
├── scripts/
│   ├── collect_data.py         # Gaussian calculation for 50 molecules
│   ├── extract_features.py     # .out → bond_features.csv
│   ├── train_model.py          # model training
│   ├── benchmark_new_mols.py   # benchmark on unseen molecules
│   ├── benchmark_acetylene.py  # acetylene case study
│   ├── benchmark.py            # condition-based comparison (rdkit/ml)
│   └── pipeline.py             # SMILES → .com CLI
├── notebooks/
│   └── pipeline_comparison.ipynb  # interactive pipeline comparison
├── figures/                    # generated plots and CSV (01~07)
├── data/                       # training data (gitignored)
└── models/                     # trained models (gitignored)
```

---

## Environment

- Python 3.11, RDKit, scikit-learn ≥ 1.4
- Gaussian 09W (Windows, `C:\G09W\g09.exe`)
- Calculation level: B3LYP/6-31G(d), `#p opt freq`
