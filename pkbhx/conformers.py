import os
import time
import tempfile
from typing import Optional
from .checkpoint import CheckpointManager

def _cleanup_auto3d_logging():
    import logging
    for name in list(logging.Logger.manager.loggerDict):
        if "auto3d" in name.lower() or "Auto3D" in name:
            logger = logging.getLogger(name)
            for h in logger.handlers[:]:
                try:
                    h.close()
                except Exception:
                    pass
                logger.removeHandler(h)

def _rdkit_conformer(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol

def generate_conformer(smiles: str, ckpt: Optional[CheckpointManager] = None, n_conformers: int = 1):
    from rdkit import Chem

    if ckpt:
        mol = ckpt.load_geometry(smiles)
        if mol is not None:
            return mol

    k = max(1, n_conformers)
    mol = None

    try:
        import torch
        from Auto3D import Auto3DOptions, main as auto3d_main
    except ImportError as e:
        # Avoid UnboundLocalError by explicitly undefining them or catching it later
        pass

    try:
        _cleanup_auto3d_logging()

        print(f"  -> Generating conformer{'s' if k > 1 else ''} (Auto3D/AIMNet2, k={k})...", end="", flush=True)
        t0 = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.smi")
            with open(inp, "w") as f:
                f.write(f"{smiles} mol\n")

            opts = Auto3DOptions(
                inp, k=k, optimizing_engine="AIMNET",
                use_gpu=torch.cuda.is_available(), verbose=False,
            )
            out = auto3d_main(opts)

            if out is not None and os.path.isfile(str(out)):
                suppl = list(Chem.SDMolSupplier(str(out), removeHs=False))
                mols = [m for m in suppl if m is not None]
            else:
                mols = []

            if mols:
                if k > 1 and len(mols) > 1:
                    best, best_e = mols[0], float("inf")
                    for m in mols:
                        try:
                            e = float(
                                m.GetProp("Energy") if m.HasProp("Energy")
                                else m.GetProp("E_tot") if m.HasProp("E_tot")
                                else "inf")
                            if e < best_e:
                                best_e, best = e, m
                        except (ValueError, KeyError):
                            pass
                    mol = best
                else:
                    mol = mols[0]
                print(f" Done ({time.time() - t0:.1f}s, {len(mols)} conf)")
            else:
                print(" No output.", end="")
                raise RuntimeError("Auto3D produced no conformers")

    except Exception as e:
        msg = str(e)
        if len(msg) > 80:
            msg = msg[:77] + "..."
        print(f" Auto3D failed ({type(e).__name__}: {msg}), using RDKit/MMFF...", end="", flush=True)
        t0 = time.time()
        mol = _rdkit_conformer(smiles)
        print(f" Done ({time.time() - t0:.1f}s)")
    finally:
        _cleanup_auto3d_logging()

    if mol is None:
        raise RuntimeError(f"All conformer methods failed for {smiles}")

    if ckpt:
        ckpt.save_geometry(smiles, mol)
    return mol
