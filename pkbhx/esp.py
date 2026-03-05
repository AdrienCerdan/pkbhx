import time
import os
import numpy as np
from abc import ABC, abstractmethod

BOHR2ANG = 0.529177210903

class ESPEngine(ABC):
    @abstractmethod
    def eval_grid(self, grid_ang: np.ndarray) -> np.ndarray:
        ...

    def eval_point(self, coord_ang: np.ndarray) -> float:
        return self.eval_grid(coord_ang.reshape(1, 3))[0]

class Psi4ESP(ESPEngine):
    def __init__(self, mol, smiles: str = "", ckpt=None,
                 method="r2scan-3c", memory="8 GB", nthreads=0):
        import psi4
        from rdkit import Chem

        if nthreads <= 0:
            nthreads = os.cpu_count() or 4
        psi4.set_num_threads(nthreads)
        psi4.set_memory(memory)
        psi4.core.clean()
        psi4.core.set_output_file("psi4_output.dat", False)

        psi4.set_options({
            "dft_radial_points": 99,
            "dft_spherical_points": 590,
            "dft_pruning_scheme": "robust",
            "dft_nuclear_scheme": "stratmann",
            "ints_tolerance": 1e-14,
            "scf_type": "df",
            "level_shift": 0.1,
        })

        BOHR2ANG_ = psi4.constants.bohr2angstroms
        wfn, saved_geom = None, None
        if ckpt and smiles:
            wfn, saved_geom = ckpt.load_wfn(smiles, engine="psi4")

        if wfn is None:
            conf = mol.GetConformer(0)
            xyz = f"{Chem.GetFormalCharge(mol)} 1\n"
            for a in mol.GetAtoms():
                p = conf.GetAtomPosition(a.GetIdx())
                xyz += f"{a.GetSymbol()} {p.x} {p.y} {p.z}\n"
            xyz += "units angstrom\nsymmetry c1\nno_reorient\nno_com"

            print(f"  -> Running DFT ({method})...", end="", flush=True)
            t0 = time.time()
            psi_mol = psi4.geometry(xyz)
            E, wfn = psi4.energy(method, return_wfn=True, molecule=psi_mol)
            print(f" Done ({time.time() - t0:.1f}s). E = {E:.5f} Ha")
            if ckpt and smiles:
                ckpt.save_wfn(smiles, wfn, engine="psi4")

        geom_bohr = saved_geom if saved_geom is not None else np.array(wfn.molecule().geometry())
        conf = mol.GetConformer(0)
        assert geom_bohr.shape[0] == mol.GetNumAtoms()
        for i in range(mol.GetNumAtoms()):
            x, y, z = geom_bohr[i] * BOHR2ANG_
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))

        self._esp = psi4.core.ESPPropCalc(wfn)
        self.wfn = wfn

    def eval_grid(self, grid_ang: np.ndarray) -> np.ndarray:
        import psi4
        mat = psi4.core.Matrix.from_array(np.ascontiguousarray(grid_ang))
        return np.array(self._esp.compute_esp_over_grid_in_memory(mat))

def create_esp_engine(engine: str, mol, smiles: str = "", ckpt=None, **kwargs) -> ESPEngine:
    if engine == "psi4":
        return Psi4ESP(mol, smiles, ckpt,
                       method=kwargs.get("method", "r2scan-3c"),
                       memory=kwargs.get("memory", "8 GB"),
                       nthreads=kwargs.get("nthreads", 0))
    else:
        raise ValueError(f"Unknown engine: {engine}. Only 'psi4' is supported in this distribution.")
