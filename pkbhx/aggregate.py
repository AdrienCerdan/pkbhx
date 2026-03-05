import numpy as np
from typing import Dict, List, Tuple
from .core import AtomResult, RegressionParams

STERIC_CORRECTION_TYPES = {"Amine"}

def compute_pkbhx(
    results: List[AtomResult],
    regression_params: Dict[str, RegressionParams],
    steric_weight: float = 1.0,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Returns {(atom_idx, site_id): (site_pk, mol_pk)}."""
    from collections import defaultdict
    type_groups: Dict[str, List[AtomResult]] = defaultdict(list)
    for r in results:
        type_groups[r.atom_type].append(r)

    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for atype, group in type_groups.items():
        model = regression_params.get(atype)
        if model is None:
            print(f"  WARNING: no regression params for '{atype}'")
            continue

        site_pks: List[Tuple[AtomResult, float]] = []
        for r in group:
            pk_raw = model.intercept + model.slope * r.vmin
            if atype in STERIC_CORRECTION_TYPES and r.occlusion > 0.05:
                correction = steric_weight * r.occlusion * abs(pk_raw)
                pk = pk_raw - correction
            else:
                pk = pk_raw
            site_pks.append((r, pk))

        mol_pk = np.log10(sum(r.degeneracy * 10**pk for r, pk in site_pks))
        for r, pk in site_pks:
            out[(r.idx, r.site_id)] = (pk, mol_pk)
    return out
