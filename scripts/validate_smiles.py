from rdkit import Chem
from collect_data import MOLECULES
bad = [(n, s) for n, s, *_ in MOLECULES if Chem.MolFromSmiles(s) is None]
print(f"총 {len(MOLECULES)}개")
print("유효하지 않은 SMILES:", bad if bad else "없음")
