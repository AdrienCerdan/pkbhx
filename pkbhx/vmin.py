import time
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

from .core import AtomResult, VDW_RADII
from .utils import _pt3d, _fibonacci_sphere

def compute_steric_occlusion(
    mol, vmin_coord: np.ndarray, acceptor_idx: int,
    n_rays: int = 200, probe_radius: float = 1.20,
) -> float:
    conf = mol.GetConformer(0)
    center = _pt3d(conf.GetAtomPosition(acceptor_idx))

    approach_dir = vmin_coord - center
    approach_norm = np.linalg.norm(approach_dir)
    if approach_norm < 1e-8:
        return 0.0
    approach_dir /= approach_norm

    acceptor_atom = mol.GetAtomWithIdx(acceptor_idx)
    exclude = {acceptor_idx}
    for nbr in acceptor_atom.GetNeighbors():
        exclude.add(nbr.GetIdx())

    block_centers = []
    block_radii = []
    for i in range(mol.GetNumAtoms()):
        if i in exclude:
            continue
        c = _pt3d(conf.GetAtomPosition(i))
        r = VDW_RADII.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 1.70)
        block_centers.append(c)
        block_radii.append(r + probe_radius)

    if not block_centers:
        return 0.0

    block_centers = np.array(block_centers)
    block_radii = np.array(block_radii)

    sphere = _fibonacci_sphere(n_rays * 2)
    dots = sphere @ approach_dir
    hemi = sphere[dots > 0.0]
    if len(hemi) == 0:
        return 0.0

    blocked = 0
    for ray_dir in hemi:
        origin = vmin_coord
        for j in range(len(block_centers)):
            oc = origin - block_centers[j]
            b = np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - block_radii[j] ** 2
            discriminant = b * b - c
            if discriminant > 0:
                t = -b - np.sqrt(discriminant)
                if t > 0.01:
                    blocked += 1
                    break

    return blocked / len(hemi)


def _compute_lone_pair_seeds(mol, conf, idx: int) -> Tuple[List[np.ndarray], int]:
    center = _pt3d(conf.GetAtomPosition(idx))
    atom = mol.GetAtomWithIdx(idx)
    neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 0]
    n_nbrs = len(neighbors)

    nbr_vecs = []
    for nbr in neighbors:
        v = _pt3d(conf.GetAtomPosition(nbr.GetIdx())) - center
        n = np.linalg.norm(v)
        if n > 1e-6:
            nbr_vecs.append(v / n)

    if not nbr_vecs:
        return [], 1

    directions: List[np.ndarray] = []
    expected = 1

    if n_nbrs == 1:
        bond = mol.GetBondBetweenAtoms(idx, neighbors[0].GetIdx())
        bt = bond.GetBondTypeAsDouble() if bond is not None else 0
        is_triple = bt >= 2.5
        is_double = (bond is not None and not is_triple and
                     (bt >= 2.0
                      or bond.GetIsAromatic()
                      or atom.GetAtomicNum() == 8 and neighbors[0].GetDegree() >= 3))
        
        if is_double:
            anti_bond = -nbr_vecs[0]
            nbr_atom = neighbors[0]
            nbr_pos = _pt3d(conf.GetAtomPosition(nbr_atom.GetIdx()))

            plane_ref = None
            for other in nbr_atom.GetNeighbors():
                if other.GetIdx() == idx:
                    continue
                v = _pt3d(conf.GetAtomPosition(other.GetIdx())) - nbr_pos
                vn = np.linalg.norm(v)
                if vn > 1e-6:
                    plane_ref = v / vn
                    break

            if plane_ref is not None:
                plane_normal = np.cross(nbr_vecs[0], plane_ref)
                pn = np.linalg.norm(plane_normal)
                if pn > 1e-6:
                    plane_normal /= pn
                    for sign in (+1, -1):
                        rot = Rotation.from_rotvec(
                            sign * np.radians(60) * plane_normal)
                        directions.append(rot.apply(anti_bond))
                    expected = 2
                else:
                    directions.append(anti_bond)
            else:
                directions.append(anti_bond)
        else:
            directions.append(-nbr_vecs[0])
            expected = 1

    elif n_nbrs == 2:
        v1, v2 = nbr_vecs[0], nbr_vecs[1]
        bisector = v1 + v2
        bis_norm = np.linalg.norm(bisector)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))

        if bis_norm < 1e-6:
            perp = np.cross(v1, [0, 0, 1])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(v1, [0, 1, 0])
            directions.append(perp / np.linalg.norm(perp))
            expected = 1
        else:
            anti_bisector = -bisector / bis_norm
            plane_normal = np.cross(v1, v2)
            pn = np.linalg.norm(plane_normal)

            if atom.GetIsAromatic() and pn > 1e-6:
                directions.append(anti_bisector)
                expected = 1
            elif angle > np.radians(105) and pn > 1e-6:
                plane_normal /= pn
                for sign in (+1, -1):
                    rot = Rotation.from_rotvec(
                        sign * np.radians(60) * plane_normal)
                    directions.append(rot.apply(anti_bisector))
                expected = 2
            elif pn > 1e-6:
                plane_normal /= pn
                tet = np.radians(54.75)
                for sign in (+1, -1):
                    lp = (anti_bisector * np.cos(tet)
                          + sign * plane_normal * np.sin(tet))
                    directions.append(lp / np.linalg.norm(lp))
                expected = 2
            else:
                directions.append(anti_bisector)
                expected = 1

    elif n_nbrs == 3:
        centroid = sum(nbr_vecs) / len(nbr_vecs)
        cn = np.linalg.norm(centroid)
        if cn > 1e-6:
            directions.append(-centroid / cn)
        else:
            pn = np.cross(nbr_vecs[0], nbr_vecs[1])
            if np.linalg.norm(pn) > 1e-6:
                directions.append(pn / np.linalg.norm(pn))
        expected = 1

    directions = [d / np.linalg.norm(d) for d in directions
                  if np.linalg.norm(d) > 1e-6]
    return directions, expected


def find_vmin_per_atom(
    mol, esp_engine, atom_types: Dict[int, str],
    grid_density: int = 250, merge_threshold: float = 0.3, steric: bool = True
) -> List[AtomResult]:
    print("  -> Calculating Vmin (Surrogate + QM Polish)...", end="", flush=True)
    t0 = time.time()
    conf = mol.GetConformer(0)
    
    base_sphere = _fibonacci_sphere(grid_density)
    coarse_pts, coarse_map = [], []
    atom_expected_n = {}

    for idx in atom_types:
        center = _pt3d(conf.GetAtomPosition(idx))
        radius = VDW_RADII.get(mol.GetAtomWithIdx(idx).GetAtomicNum(), 1.55)
        
        lp_dirs, expected_n = _compute_lone_pair_seeds(mol, conf, idx)
        atom_expected_n[idx] = expected_n

        for d in lp_dirs:
            coarse_pts.append(center + d * radius)
            coarse_map.append(idx)
        for pt in (center + base_sphere * radius):
            coarse_pts.append(pt)
            coarse_map.append(idx)

    coarse_pts = np.array(coarse_pts)
    coarse_vals = esp_engine.eval_grid(coarse_pts)

    candidate_seeds = []
    for idx in atom_types:
        idx_mask = [m == idx for m in coarse_map]
        pts_i, vals_i = coarse_pts[idx_mask], coarse_vals[idx_mask]
        
        order = np.argsort(vals_i)
        seeds_for_atom = []
        for si in order:
            pt = pts_i[si]
            if all(np.linalg.norm(pt - s) > merge_threshold for s in seeds_for_atom):
                seeds_for_atom.append(pt)
            if len(seeds_for_atom) >= atom_expected_n[idx] + 1:
                break
        for s in seeds_for_atom:
            candidate_seeds.append((idx, s))

    grid_dim = 7
    spacing = 0.1
    offsets = np.linspace(-(grid_dim // 2) * spacing, (grid_dim // 2) * spacing, grid_dim)
    
    dx, dy, dz = np.meshgrid(offsets, offsets, offsets, indexing='ij')
    local_mesh_template = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=-1)
    
    all_micro_pts = []
    for _, seed in candidate_seeds:
        all_micro_pts.append(local_mesh_template + seed)
        
    all_micro_pts = np.vstack(all_micro_pts)
    all_micro_vals = esp_engine.eval_grid(all_micro_pts)

    results: List[AtomResult] = []
    raw_results = defaultdict(list)
    pts_per_grid = grid_dim ** 3

    for i, (idx, seed) in enumerate(candidate_seeds):
        start_idx = i * pts_per_grid
        end_idx = start_idx + pts_per_grid
        local_vals = all_micro_vals[start_idx:end_idx].reshape(grid_dim, grid_dim, grid_dim)
        
        x_axes, y_axes, z_axes = seed[0] + offsets, seed[1] + offsets, seed[2] + offsets
        surrogate = RegularGridInterpolator(
            (x_axes, y_axes, z_axes), local_vals, method='cubic', bounds_error=True
        )
        
        try:
            surr_res = minimize(
                lambda c: surrogate(c)[0],
                seed, method="L-BFGS-B", tol=1e-5,
                bounds=[(x_axes[0], x_axes[-1]), (y_axes[0], y_axes[-1]), (z_axes[0], z_axes[-1])]
            )
            val, coord = surr_res.fun, surr_res.x
        except ValueError:
            best_idx = np.argmin(local_vals)
            val, coord = np.min(local_vals), all_micro_pts[start_idx:end_idx][best_idx]

        polish_bounds = [
            (coord[0] - 0.05, coord[0] + 0.05),
            (coord[1] - 0.05, coord[1] + 0.05),
            (coord[2] - 0.05, coord[2] + 0.05)
        ]
        
        try:
            qm_res = minimize(
                lambda c: esp_engine.eval_point(c),
                coord, 
                method="L-BFGS-B", 
                bounds=polish_bounds,
                options={
                    'maxiter': 5,
                    'maxcor': 3,
                    'ftol': 1e-6,      
                    'gtol': 1e-4       
                }
            )
            val, coord = qm_res.fun, qm_res.x
        except Exception:
            pass

        center = _pt3d(conf.GetAtomPosition(idx))
        radius = VDW_RADII.get(mol.GetAtomWithIdx(idx).GetAtomicNum(), 1.55)
        d = np.linalg.norm(coord - center)
        
        if (0.7 * radius) < d < (1.5 * radius):
            raw_results[idx].append((val, coord))

    for idx, raw_list in raw_results.items():
        atom = mol.GetAtomWithIdx(idx)
        raw_list.sort(key=lambda x: x[0])
        
        unique = []
        for val, coord in raw_list:
            if all(np.linalg.norm(coord - uc) > merge_threshold for _, uc in unique):
                unique.append((val, coord))
            if len(unique) >= atom_expected_n[idx]:
                break

        center = _pt3d(conf.GetAtomPosition(idx))
        for sid, (val, coord) in enumerate(unique):
            occ = compute_steric_occlusion(mol, coord, idx) if steric else 0.0
            results.append(AtomResult(
                idx=idx, symbol=atom.GetSymbol(), atom_type=atom_types[idx],
                vmin=val, coords=coord, site_id=sid,
                dist=float(np.linalg.norm(coord - center)), occlusion=occ,
                degeneracy=3 if atom_types[idx] in ("Fluorine", "Chlorine") else 1
            ))

    print(f" Done ({time.time() - t0:.1f}s)")
    return results
