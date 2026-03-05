from typing import Dict

def identify_acceptors(mol) -> Dict[int, str]:
    """Return {atom_idx: functional_group_type} for all HBA sites."""
    from rdkit import Chem

    PATTERNS = [
        # --- Generic ---
        ("Amine",         "[#7;!a;!$([#7]~[#6](~[#8])~[#7,#8]);!$([#7]S(=O)=O);!$([#7]=[#6]);!$([#7]#*)]", "N"),
        ("Ether/Alcohol", "[OX2;!a]", "O"),
        ("Aromatic O",    "[oX2]", "O"),
        ("Fluorine",      "[#9]", "F"),
        ("Chlorine",      "[#17]", "Cl"),
        ("Divalent S",    "[#16X2]", "S"),

        # --- Specific nitrogen ---
        ("Aromatic N",    "[nD2H0]", "N"),
        ("Imine",         "[NX2]=[CX3]", "N"),
        ("Nitrile",       "[NX1]#[CX2]", "N"),

        # --- Specific oxygen ---
        ("Carbonyl",      "[#8X1;!$([#8]~[#16,#34]);!$([#8]~[#15,#33]);!$([#8-]~[#7+])]~[#6X3]", "O"),
        ("Thiocarbonyl",  "[#16X1]~[#6X3]", "S"),
        ("Chalcogen oxide", "[#8X1]~[#16,#34]", "O"),
        ("Pnictogen oxide", "[#8X1]~[#15,#33]", "O"),
        ("N-oxide",       "[OX1-][#7+]", "O"),
    ]
    TARGET_Z = {"N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}

    atom_types: Dict[int, str] = {}
    for type_name, smarts, target_elem in PATTERNS:
        qmol = Chem.MolFromSmarts(smarts)
        if qmol is None:
            print(f"  WARNING: invalid SMARTS for '{type_name}': {smarts}")
            continue
        tz = TARGET_Z[target_elem]
        for match in mol.GetSubstructMatches(qmol):
            for idx in match:
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == tz:
                    atom_types[idx] = type_name
    return atom_types
