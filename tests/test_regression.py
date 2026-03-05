import pytest
import numpy as np

try:
    from rdkit import Chem
except ImportError:
    Chem = None

from pkbhx.acceptors import identify_acceptors
from pkbhx.core import AtomResult, REGRESSION_PARAMS_R2SCAN
from pkbhx.aggregate import compute_pkbhx


@pytest.mark.skipif(Chem is None, reason="RDKit is required for this test")
def test_identify_acceptors():
    """Regression test for the SMARTS-based acceptor identification logic."""
    # Test pyridine (Aromatic N)
    mol = Chem.MolFromSmiles("c1ccncc1")
    acceptors = identify_acceptors(mol)
    assert len(acceptors) == 1
    n_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'][0]
    assert acceptors[n_idx] == "Aromatic N"

    # Test Acetone (Carbonyl)
    mol = Chem.MolFromSmiles("CC(=O)C")
    acceptors = identify_acceptors(mol)
    assert len(acceptors) == 1
    o_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'][0]
    assert acceptors[o_idx] == "Carbonyl"

    # Test Ethanol (Ether/Alcohol)
    mol = Chem.MolFromSmiles("CCO")
    acceptors = identify_acceptors(mol)
    assert len(acceptors) == 1
    o_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'][0]
    assert acceptors[o_idx] == "Ether/Alcohol"

    # Test Acetonitrile (Nitrile)
    mol = Chem.MolFromSmiles("CC#N")
    acceptors = identify_acceptors(mol)
    assert len(acceptors) == 1
    n_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'][0]
    assert acceptors[n_idx] == "Nitrile"


def test_compute_pkbhx():
    """Regression test for the mathematical pKBHX aggregation and scaling."""
    # Mock an Aromatic N result
    # For Aromatic N: intercept=-3.1376, slope=-52.8126
    # With vmin = -0.05, pk_raw = -3.1376 + (-52.8126 * -0.05) = -0.49697
    
    mock_results = [
        AtomResult(
            idx=0,
            symbol="N",
            atom_type="Aromatic N",
            vmin=-0.05,
            coords=np.array([0., 0., 0.]),
            site_id=0,
            dist=1.5,
            occlusion=0.0,
            degeneracy=1
        )
    ]
    
    pkbhx = compute_pkbhx(mock_results, REGRESSION_PARAMS_R2SCAN, steric_weight=1.0)
    
    assert (0, 0) in pkbhx
    site_pk, mol_pk = pkbhx[(0, 0)]
    
    expected_pk = -3.1376 + (-52.8126 * -0.05)
    
    np.testing.assert_allclose(site_pk, expected_pk, rtol=1e-5)
    np.testing.assert_allclose(mol_pk, expected_pk, rtol=1e-5)


def test_amine_steric_correction():
    """Regression test to ensure steric penalties apply to bulkier Amines."""
    # For Amine: intercept=-1.4884, slope=-34.4386
    # vmin = -0.05 -> raw_pk = -1.4884 + (-34.4386 * -0.05) = 0.23353
    # With 50% occlusion (0.5), correction = 1.0 * 0.5 * abs(0.23353) = 0.116765
    # final_pk = 0.23353 - 0.116765 = 0.116765

    mock_results = [
        AtomResult(
            idx=1,
            symbol="N",
            atom_type="Amine",
            vmin=-0.05,
            coords=np.array([1., 1., 1.]),
            site_id=0,
            dist=1.5,
            occlusion=0.5, # 50% occlusion
            degeneracy=1
        )
    ]

    pkbhx = compute_pkbhx(mock_results, REGRESSION_PARAMS_R2SCAN, steric_weight=1.0)
    
    site_pk, _ = pkbhx[(1, 0)]
    
    expected_raw_pk = -1.4884 + (-34.4386 * -0.05)
    expected_correction = 1.0 * 0.5 * abs(expected_raw_pk)
    expected_final_pk = expected_raw_pk - expected_correction
    
    np.testing.assert_allclose(site_pk, expected_final_pk, rtol=1e-5)
    assert site_pk < expected_raw_pk, "Steric penalty should reduce basicity"
