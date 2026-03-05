import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional

BOHR2ANG = 0.529177210903

VDW_RADII = {
    1: 1.20,  6: 1.70,  7: 1.55,  8: 1.52,   9: 1.47,
    14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 34: 1.90,
    35: 1.85, 53: 1.98,
}

EXPECTED_LP_COUNT = {
    "sp_terminal":  1,
    "sp2_terminal": 2,
    "sp2_2nbr":     2,
    "sp3_2nbr":     2,
    "sp2_ring":     1,
    "sp3_3nbr":     1,
}

@dataclass
class AtomResult:
    idx: int
    symbol: str
    atom_type: str
    vmin: float            # Hartree/e
    coords: np.ndarray     # Å
    site_id: int = 0
    dist: float = 0.0      # Å from nucleus
    occlusion: float = 0.0 # fraction of hemisphere blocked [0, 1]
    degeneracy: int = 1

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "coords"}
        d["coords"] = self.coords.tolist()
        return d

@dataclass
class RegressionParams:
    slope: float
    intercept: float
    rmse: float
    count: int

REGRESSION_PARAMS_R2SCAN: Dict[str, RegressionParams] = {
    "Amine":           RegressionParams(-34.4386, -1.4884, 0.324, 171),
    "Aromatic N":      RegressionParams(-52.8126, -3.1376, 0.150, 71),
    "Imine":           RegressionParams(-48.4007, -2.3309, 0.236, 28),
    "Nitrile":         RegressionParams(-50.1167, -3.2273, 0.198, 28),
    "N-oxide":         RegressionParams(-74.3261, -4.4159, 0.589, 16),
    "Chalcogen oxide": RegressionParams(-47.7009, -2.2794, 0.224, 17),
    "Pnictogen oxide": RegressionParams(-61.1141, -3.3839, 0.549, 16),
    "Carbonyl":        RegressionParams(-57.2911, -3.5271, 0.208, 128),
    "Ether/Alcohol":   RegressionParams(-35.9245, -2.0338, 0.239, 99),
    "Aromatic O":      RegressionParams(-35.9245, -2.0338, 0.158, 11),
    "Thiocarbonyl":    RegressionParams(-51.8837, -2.2649, 0.384, 10),
    "Divalent S":      RegressionParams(-39.1666, -2.1243, 0.127, 17),
    "Fluorine":        RegressionParams(-16.4441, -1.2540, 0.276, 23),
}

ENGINE_PARAMS = {
    "psi4": REGRESSION_PARAMS_R2SCAN,
}
