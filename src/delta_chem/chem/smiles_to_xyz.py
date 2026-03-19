"""
SMILES → RDKit ETKDG → XYZ file
"""
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_xyz(smiles: str, xyz_path: str, mol_name: str = "molecule") -> None:
    """Convert SMILES to a 3D XYZ file using RDKit ETKDG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        raise RuntimeError("RDKit failed to generate 3D coordinates.")

    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]

    with open(xyz_path, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"{mol_name}\n")
        for atom, pos in zip(atoms, conf.GetPositions()):
            f.write(f"{atom.GetSymbol():2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}\n")
