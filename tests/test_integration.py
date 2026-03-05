import pytest
import numpy as np

try:
    from rdkit import Chem
except ImportError:
    Chem = None

from pkbhx.conformers import generate_conformer
from pkbhx.esp import create_esp_engine


@pytest.mark.skipif(Chem is None, reason="RDKit is required for this test")
def test_generate_conformer_auto3d():
    """Integration test for Auto3D conformer generation (or RDKit fallback)."""
    smiles = "c1ccncc1"
    
    # generate_conformer should return an RDKit Mol object
    mol = generate_conformer(smiles, ckpt=None, n_conformers=1)
    
    assert mol is not None
    assert isinstance(mol, Chem.Mol)
    
    # Pyridine should have 11 atoms (5C, 1N, 5H)
    assert mol.GetNumAtoms() == 11
    
    # Ensure it has 3D coordinates
    conf = mol.GetConformer()
    assert conf.Is3D()


@pytest.mark.skipif(Chem is None, reason="RDKit is required for this test")
def test_psi4_esp_engine():
    """Integration test for Psi4 ESP Engine."""
    smiles = "O" # Water is fast to compute
    mol = generate_conformer(smiles, ckpt=None, n_conformers=1)
    
    # We use a very cheap method for the test instead of r2scan-3c
    esp_engine = create_esp_engine(
        engine="psi4", 
        mol=mol, 
        smiles=smiles, 
        ckpt=None, 
        method="hf/sto-3g", # Cheap method for fast test execution
        memory="1 GB", 
        nthreads=1
    )
    
    assert esp_engine is not None
    
    # Ensure it can compute the ESP at a given 3D coordinate (e.g. oxygen's position)
    oxy_pos = mol.GetConformer().GetAtomPosition(0)
    coords = np.array([[oxy_pos.x, oxy_pos.y, oxy_pos.z]])
    
    # Calculate ESP at the oxygen nucleus (should be a valid float)
    esp_values = esp_engine.eval_grid(coords)
    
    assert len(esp_values) == 1
    assert isinstance(esp_values[0], float)
