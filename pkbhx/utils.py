import numpy as np

def _pt3d(pt) -> np.ndarray:
    """RDKit Point3D -> numpy [x, y, z]."""
    return np.array([pt.x, pt.y, pt.z])

def _fibonacci_sphere(n: int = 200) -> np.ndarray:
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n, dtype=float)
    y = 1.0 - (i / (n - 1)) * 2.0
    r = np.sqrt(1.0 - y * y)
    theta = phi * i
    return np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])

def _canonical_smiles(smi: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    return Chem.MolToSmiles(mol)

def generate_mapped_smiles(mol) -> str:
    from rdkit import Chem
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        if atom.GetAtomicNum() > 1:
            atom.SetAtomMapNum(atom.GetIdx())
    mol_clean = Chem.RemoveHs(mol_copy)
    return Chem.MolToSmiles(mol_clean)
