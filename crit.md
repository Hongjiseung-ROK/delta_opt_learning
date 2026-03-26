# 현 상황 비판적 분석 (2026-03-26)

## 1. 프로젝트 현황 요약

| 항목 | 현황 |
|------|------|
| 학습 데이터 | 120개 분자, 1,750개 결합 (FreeSolv 일부 포함) |
| 현재 모델 | GradientBoostingRegressor (delta mode) |
| CV MAE (50mol) | 0.0022 ± 0.0003 Å |
| CV MAE (120mol) | 0.0029 ± 0.0002 Å ← 데이터 추가에도 **악화** |
| 목표 | Gaussian DFT 최적화 스텝 수 감소 |
| 벤치마킹 결과 | acrolein: ML-120이 MMFF보다 **오히려 스텝 증가** |

---

## 2. 핵심 문제점

### 2.1 구조 왜곡 (가장 심각)

**문제**: `corrector.py`의 DFS 순회 방식은 결합을 하나씩 순차적으로 스케일링한다.

```
parent → child: new_coords[child] = new_coords[parent] + (child - parent) * scale
```

- 결합 A-B를 먼저 늘리면, B에 붙은 모든 자식 원자들이 함께 이동
- 다음 결합 B-C를 스케일할 때 이미 B 위치가 이동되어 있음
- **결과**: 결합각 MAE 0.73°, 이면각 MAE 2.45°로 측정됨

**근본 원인**: 각 결합 길이를 독립적으로 최적화하는 방식 자체의 한계.
분자는 3D 구조이며, 결합 길이 보정은 결합각/이면각과 **결합되어(coupled)** 있다.

**영향**: acrolein(공액 C=C-C=O)처럼 결합각에 민감한 분자에서 DFT 초기구조가
MMFF보다 오히려 더 먼 지점에서 출발해 수렴 스텝이 증가.

---

### 2.2 Feature 중복성

Feature ablation (120mol 데이터, 5 실험) 결과:

| 실험 | CV MAE (Å) | 비고 |
|------|------------|------|
| 현재 전체 10개 | 0.0031 | hybridization + ring 포함 |
| hybridization 제거 (각도로 대체) | 0.0028 | 개선 |
| 핵심 6개 (ring 관련 제거) | 0.0030 | 약간 악화 |
| 각도 없는 구 방식 | 0.0035 | 각도가 중요함을 반증 |
| **이웃 문맥 추가** | **0.0027** | 현재 최선 |

핵심 발견:
- `hybridization` ↔ `mmff_angle`의 상관관계 r=0.80 → 정보 중복
- `is_in_ring` ↔ `ring_size` 상관관계 r=1.00 → 완전 중복
- `mmff_angle` feature가 hybridization을 내포하며 더 정확한 기하 정보 제공

---

### 2.3 데이터 확장 역효과

120mol 데이터가 50mol보다 **CV MAE 악화** (0.0022 → 0.0029):

- FreeSolv 분자들은 Cl, S, Br, P, F 등 다양한 원소를 포함
- 데이터가 늘었지만 **원소 종류 다양화** → 결합 유형별 데이터 희소성 증가
- 현재 모델은 원소 유형을 OrdinalEncoder로 처리 → 원소 간 화학적 관계 무시

---

### 2.4 모델 구조적 한계

GradientBoostingRegressor 기반 현재 접근의 근본 한계:

1. **결합 독립성 가정**: 각 결합을 완전히 독립적으로 예측
   - 실제로는 인접 결합들이 서로 영향을 주고받음 (공액, 하이퍼컨쥬게이션 등)
   - 분자 전체 맥락을 고려한 예측 불가

2. **좌표 보정 방식 한계**: 결합 길이 보정 → 3D 좌표 재구성 단계에서 정보 손실
   - 결합 길이만 보정하면 결합각/이면각은 MMFF 값 그대로 유지되지 않음
   - DFS 순회 방식이 전파 오류를 발생시킴

3. **Feature 수동 설계**: mmff_angle, ring_size 등 수동으로 설계한 feature
   - 분자 화학의 복잡성을 충분히 표현하기 어려움
   - 장거리 전자 효과 (공명, 유도 효과) 포착 불가

---

## 3. GNN 접근법의 잠재력

### 3.1 왜 GNN인가

| 비교 항목 | GBM (현재) | GNN (제안) |
|----------|-----------|-----------|
| 분자 표현 | 결합 단위 독립 feature | 그래프 전체 메시지 패싱 |
| 이웃 정보 | 수동 feature 계산 | 자동 집계 (k-hop 이웃) |
| 원소 표현 | OrdinalEncoder (순서 없는 인코딩) | 학습 가능한 원소 임베딩 |
| 결합 간 커플링 | 없음 | 메시지 패싱으로 자연스럽게 포착 |
| 장거리 효과 | 불가 | 다중 레이어로 확장 가능 |

### 3.2 제안 아키텍처: Edge-Conditioned MPNN

```
입력:
  - 노드(원자): 원소 종류, hybridization, 전하
  - 엣지(결합): bond_order, is_in_ring, ring_size, mmff_length, mmff_angle_1, mmff_angle_2

메시지 패싱 (3 레이어):
  h_i^(l+1) = GRU(h_i^(l), Σ_{j∈N(i)} MLP(h_j^(l) || e_ij))

엣지 레벨 예측:
  pred_delta_ij = MLP_out(h_i^(L) || h_j^(L) || e_ij)

타겟: dft_length - mmff_length (delta mode)
```

### 3.3 좌표 보정 방식 개선 (장기 목표)

현재 DFS 방식 대신 **제약 최적화** 접근:

```
minimize  Σ_bonds ||d_pred - d_current||²
subject to  각도/이면각 보존 (Lagrange multiplier)
```

또는 GNN이 직접 **3D 좌표 delta**를 예측 (EquivariantGNN, e.g., SchNet, DimeNet):
- 회전/이동 등변성(equivariance) 보장
- 결합 길이 + 결합각 + 이면각 동시 보정

---

## 4. 즉각 개선 가능 항목

### 단기 (현재 GBM 유지)

1. **Feature 정리**: hybridization 제거, `mmff_angle_1/2` + 이웃 문맥 feature 사용
2. **corrector.py 개선**: DFS 대신 `rdkit.AllChem.SetBondLength()`를 사용하여 결합 길이 직접 설정
   - RDKit 내장 함수는 결합각 보존을 시도함
3. **데이터 전략**: 원소 유형별 균형 있는 데이터 수집 (C/H/O/N 집중)

### 중기 (GNN 도입)

1. `scripts/gnn/` 폴더에 MPNN 구현
2. 기존 GBM + corrector.py와 동일한 벤치마킹 인프라에서 비교
3. 좌표 보정: AllChem.SetBondLength 기반으로 교체

### 장기 (근본 해결)

1. 3D 좌표 직접 예측 (Equivariant GNN)
2. 결합 길이만이 아닌 전체 분자 형태 보정
3. 더 큰 DFT 데이터셋 구축 (QM9 등)

---

## 5. 결론

**현재 파이프라인의 성능 병목은 결합 길이 예측 정확도가 아니라 3D 좌표 재구성 방식이다.**

- 예측 정확도(CV MAE 0.002Å)는 충분히 좋음
- 문제는 결합 길이 보정을 3D 좌표에 반영하는 DFS 방식이 구조를 왜곡함
- GNN 도입과 함께 RDKit `SetBondLength` 기반 좌표 보정으로 전환해야 함

우선순위:
1. **corrector.py를 SetBondLength 방식으로 교체** (낮은 위험, 즉각 개선 기대)
2. **GNN 모델 구현 및 비교**
3. **Equivariant GNN으로의 장기 발전**
