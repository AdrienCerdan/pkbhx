import numpy as np
import hashlib
import os

class CheckpointManager:
    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _hash(self, smiles: str) -> str:
        return hashlib.md5(smiles.encode()).hexdigest()

    def save_geometry(self, smiles, mol):
        from rdkit import Chem
        path = os.path.join(self.base_dir, f"{self._hash(smiles)}_geom.sdf")
        w = Chem.SDWriter(path)
        w.write(mol)
        w.close()
        print(f"  [Cache] Geometry saved")

    def load_geometry(self, smiles):
        from rdkit import Chem
        path = os.path.join(self.base_dir, f"{self._hash(smiles)}_geom.sdf")
        if not os.path.exists(path):
            return None
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mol = next(suppl, None)
        if mol is not None:
            print(f"  [Cache] Geometry loaded")
        return mol

    def save_wfn(self, smiles, wfn, engine="psi4"):
        if engine == "psi4":
            import psi4
            base = os.path.join(self.base_dir, f"{self._hash(smiles)}_wfn")
            wfn.to_file(base)
            np.save(base + "_geom_bohr.npy", np.array(wfn.molecule().geometry()))
            print(f"  [Cache] Wavefunction saved")

    def load_wfn(self, smiles, engine="psi4"):
        if engine == "psi4":
            import psi4
            base = os.path.join(self.base_dir, f"{self._hash(smiles)}_wfn")
            npy = base + ".npy"
            if not os.path.exists(npy):
                return None, None
            try:
                wfn = psi4.core.Wavefunction.from_file(npy)
                geom_path = base + "_geom_bohr.npy"
                geom = np.load(geom_path) if os.path.exists(geom_path) else None
                print(f"  [Cache] Wavefunction loaded")
                return wfn, geom
            except Exception as e:
                print(f"  [Cache] Wfn load failed: {e}")
                return None, None
        return None, None
