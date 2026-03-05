import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .core import AtomResult, RegressionParams
from .utils import _pt3d

def write_pdb(mol, results: List[AtomResult], filename: str = "vmin_sites.pdb"):
    conf = mol.GetConformer(0)
    with open(filename, "w") as f:
        f.write(f"REMARK   pKBHX Vmin sites\n")
        for i, atom in enumerate(mol.GetAtoms()):
            p = _pt3d(conf.GetAtomPosition(i))
            f.write(
                f"HETATM{i+1:5d} {atom.GetSymbol():^4s} MOL A   1    "
                f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                f"  1.00  0.00           {atom.GetSymbol():>2s}\n"
            )
        n = mol.GetNumAtoms()
        for j, r in enumerate(results):
            c = r.coords
            label = f"V{r.symbol}{r.idx}".ljust(4)[:4]
            f.write(
                f"HETATM{n+j+1:5d} {label} VMN A {r.site_id+2:3d}    "
                f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}"
                f"  1.00 {r.vmin:6.2f}           HE\n"
            )
        f.write("END\n")
    print(f"  -> PDB: {filename}")

def draw_2d(mol, annotations: Dict[int, str], filename: str = "pkbhx.svg"):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    heavy_map = {}
    ni = 0
    for oi in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(oi).GetAtomicNum() != 1:
            heavy_map[oi] = ni
            ni += 1

    mol2d = Chem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol2d)
    d = rdMolDraw2D.MolDraw2DSVG(600, 600)
    d.drawOptions().addAtomIndices = True

    hl_atoms, hl_colors = [], {}
    for oidx, lbl in annotations.items():
        nidx = heavy_map.get(oidx)
        if nidx is None:
            continue
        mol2d.GetAtomWithIdx(nidx).SetProp("atomNote", lbl)
        hl_atoms.append(nidx)
        try:
            v = float(lbl)
            hl_colors[nidx] = ((1, .6, .6) if v > 2
                               else (1, .8, .6) if v > 1
                               else (.8, .8, 1))
        except ValueError:
            pass

    d.DrawMolecule(mol2d, highlightAtoms=hl_atoms, highlightAtomColors=hl_colors)
    d.FinishDrawing()
    Path(filename).write_text(d.GetDrawingText())
    print(f"  -> SVG: {filename}")

def generate_esp_cube(wfn, mol, filename="esp_surface.cube", spacing=0.3, padding=4.0):
    import psi4
    BOHR2ANG_ = psi4.constants.bohr2angstroms

    print("  -> Generating ESP cube...", end="", flush=True)
    t0 = time.time()
    conf = mol.GetConformer(0)
    coords = np.array([_pt3d(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    lo = coords.min(axis=0) - padding
    hi = coords.max(axis=0) + padding
    ns = ((hi - lo) / spacing).astype(int) + 1
    nx, ny, nz = ns

    x = np.linspace(lo[0], lo[0] + (nx - 1) * spacing, nx)
    y = np.linspace(lo[1], lo[1] + (ny - 1) * spacing, ny)
    z = np.linspace(lo[2], lo[2] + (nz - 1) * spacing, nz)

    grid = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z])
    esp = psi4.core.ESPPropCalc(wfn)
    vals = np.zeros(len(grid))
    cs = 50000
    for i in range(0, len(grid), cs):
        chunk = grid[i:i + cs]
        mat = psi4.core.Matrix.from_array(chunk)
        vals[i:i + len(chunk)] = np.array(esp.compute_esp_over_grid_in_memory(mat))

    bohr = 1.0 / BOHR2ANG_
    with open(filename, "w") as f:
        f.write("ESP — pKBHX predictor\nElectrostatic potential (Hartree/e)\n")
        na = mol.GetNumAtoms()
        f.write(f"{na:5d}{lo[0]*bohr:12.6f}{lo[1]*bohr:12.6f}{lo[2]*bohr:12.6f}\n")
        sb = spacing * bohr
        f.write(f"{nx:5d}{sb:12.6f}{0:12.6f}{0:12.6f}\n")
        f.write(f"{ny:5d}{0:12.6f}{sb:12.6f}{0:12.6f}\n")
        f.write(f"{nz:5d}{0:12.6f}{0:12.6f}{sb:12.6f}\n")
        for i in range(na):
            a = mol.GetAtomWithIdx(i)
            c = coords[i] * bohr
            f.write(f"{a.GetAtomicNum():5d}{0:12.6f}{c[0]:12.6f}{c[1]:12.6f}{c[2]:12.6f}\n")
        idx = 0
        for _ in range(nx):
            for _ in range(ny):
                line = []
                for _ in range(nz):
                    line.append(vals[idx]); idx += 1
                    if len(line) == 6:
                        f.write("".join(f"{v:13.5e}" for v in line) + "\n")
                        line = []
                if line:
                    f.write("".join(f"{v:13.5e}" for v in line) + "\n")

    dt = time.time() - t0
    print(f" Done ({dt:.1f}s) [{nx}×{ny}×{nz} = {nx*ny*nz:,} pts]")
    print(f"  -> Cube: {filename}")

def load_custom_params(path: str) -> Dict[str, RegressionParams]:
    data = json.loads(Path(path).read_text())
    params = {}
    for name, vals in data.items():
        params[name] = RegressionParams(
            slope=vals["slope"],
            intercept=vals["intercept"],
            rmse=vals.get("rmse", 0.0),
            count=vals.get("count", 0),
        )
    return params
