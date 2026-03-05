import argparse
import json
import sys
import time
from typing import List, Tuple, Optional, Dict
import numpy as np

from .utils import _canonical_smiles, generate_mapped_smiles
from .core import ENGINE_PARAMS, REGRESSION_PARAMS_R2SCAN
from .conformers import generate_conformer
from .esp import create_esp_engine
from .acceptors import identify_acceptors
from .vmin import find_vmin_per_atom
from .aggregate import compute_pkbhx
from .io import write_pdb, draw_2d, generate_esp_cube, load_custom_params
from .checkpoint import CheckpointManager

def predict_one(smiles: str, args, name: str = "", exp_pkbhx: Optional[float] = None) -> dict:
    ckpt = CheckpointManager(args.cache_dir) if not args.no_cache else None

    label = name or smiles
    print(f"\n{'='*60}")
    print(f"  Target: {label}  [engine: {args.engine}]")
    print(f"{'='*60}")

    mol = generate_conformer(smiles, ckpt, n_conformers=args.conformers)

    esp_engine = create_esp_engine(
        args.engine, mol, smiles, ckpt,
        method=args.method, memory=args.memory,
        nthreads=args.threads
    )

    atom_types = identify_acceptors(mol)
    print(f"  Acceptors: { {i: (mol.GetAtomWithIdx(i).GetSymbol(), t) for i, t in atom_types.items()} }")

    results = find_vmin_per_atom(mol, esp_engine, atom_types, steric=not args.no_steric)

    if args.params_json:
        reg_params = load_custom_params(args.params_json)
    else:
        reg_params = ENGINE_PARAMS.get(args.engine, REGRESSION_PARAMS_R2SCAN)

    pkbhx = compute_pkbhx(results, reg_params, steric_weight=args.steric_weight)

    degen_map = {(r.idx, r.site_id): r.degeneracy for r in results}
    all_site_data = [(pkbhx[k][0], degen_map.get(k, 1))
                     for k in pkbhx if pkbhx[k][0] is not None]
    global_mol_pk = (float(np.log10(sum(d * 10**pk for pk, d in all_site_data)))
                     if all_site_data else None)

    hdr = (f"{'Atom':<6} {'#':<3} {'Type':<18} {'Vmin(au)':<11} "
           f"{'SitepK':<8} {'MolpK':<8} {'d(Å)':<6} {'Occ%':<5}")
    print(f"\n{hdr}\n{'-'*len(hdr)}")
    annotations: Dict[int, str] = {}
    for r in results:
        spk, mpk = pkbhx.get((r.idx, r.site_id), (float("nan"), float("nan")))
        occ_pct = r.occlusion * 100
        print(f"{r.symbol}{r.idx:<5} {r.site_id:<3} {r.atom_type:<18} "
              f"{r.vmin:<11.5f} {spk:<8.2f} {mpk:<8.2f} "
              f"{r.dist:<6.3f} {occ_pct:4.0f}%")
        if r.idx not in annotations or r.site_id == 0:
            annotations[r.idx] = f"{spk:.2f}"

    if global_mol_pk is not None:
        exp_str = f"  exp={exp_pkbhx:.2f}" if exp_pkbhx is not None else ""
        err_str = ""
        if exp_pkbhx is not None:
            err = global_mol_pk - exp_pkbhx
            err_str = f"  err={err:+.2f}"
        print(f"\n  Molecular pKBHX (pred) = {global_mol_pk:.2f}{exp_str}{err_str}")

    prefix = (args.output_prefix or _canonical_smiles(smiles).replace("/", "").replace("\\", "")[:40])
    write_pdb(mol, results, f"{prefix}_vmin.pdb")
    draw_2d(mol, annotations, f"{prefix}_pkbhx.svg")

    if args.esp_cube:
        if args.engine == "psi4" and hasattr(esp_engine, 'wfn'):
            generate_esp_cube(esp_engine.wfn, mol, f"{prefix}_esp.cube",
                              spacing=args.esp_spacing, padding=args.esp_padding)
        else:
            print("  -> ESP cube only supported with Psi4 engine")

    site_records = []
    for r in results:
        spk, mpk = pkbhx.get((r.idx, r.site_id), (None, None))
        site_records.append({
            **r.to_dict(),
            "site_pkbhx": round(spk, 4) if spk is not None else None,
            "type_mol_pkbhx": round(mpk, 4) if mpk is not None else None,
        })

    return {
        "name": name,
        "smiles": smiles,
        "mapped_smiles": generate_mapped_smiles(mol),
        "engine": args.engine,
        "mol_pkbhx_pred": round(global_mol_pk, 4) if global_mol_pk is not None else None,
        "mol_pkbhx_exp": exp_pkbhx,
        "sites": site_records,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pkbhx-predict",
        description="Predict site-specific pKBHX via Vmin (Rowan/Kenny workflow).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "c1ccncc1"                                 # pyridine (Psi4)
  %(prog)s --csv database.csv --json results.json      # batch from CSV
  %(prog)s -f mols.smi --json results                  # batch from SMI
  %(prog)s "CCO" --conformers 5                        # multi-conformer
  %(prog)s "CCO" --params-json my_params.json           # custom regression

CSV format:  #Name,SMILES,pKHB,Reference,URL  (header line starts with #)
""",
    )
    p.add_argument("smiles", nargs="?", help="SMILES string to predict")
    p.add_argument("-f", "--file", help="File with SMILES (one per line)")
    p.add_argument("--csv", help="CSV file with columns: Name,SMILES,pKHB[,...]")

    p.add_argument("--engine", default="psi4", choices=["psi4"], help="ESP engine (default: psi4)")

    p.add_argument("--method", default=None, help="DFT method (default: r2scan-3c for psi4)")
    p.add_argument("--memory", default="8 GB", help="Memory allocation (default: 8 GB)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads, 0=auto (default: 0)")

    p.add_argument("--conformers", type=int, default=1, help="Number of conformers to generate (default: 1)")

    p.add_argument("--no-steric", action="store_true", help="Disable steric occlusion correction")
    p.add_argument("--steric-weight", type=float, default=1.0, help="Steric correction weight (default: 1.0)")

    p.add_argument("--params-json", default=None, help="Custom regression parameters JSON file")

    p.add_argument("--esp-cube", action="store_true", help="Generate ESP cube file (Psi4 only)")
    p.add_argument("--esp-spacing", type=float, default=0.3, help="Cube grid spacing Å (default: 0.3)")
    p.add_argument("--esp-padding", type=float, default=4.0, help="Cube grid padding Å (default: 4.0)")

    p.add_argument("-o", "--output-prefix", help="Output file prefix (default: from SMILES)")
    p.add_argument("--json", help="Write JSON results to this file")

    p.add_argument("--cache-dir", default="checkpoints", help="Checkpoint directory")
    p.add_argument("--no-cache", action="store_true", help="Disable checkpoint caching")

    return p

def _parse_csv(path: str) -> List[Tuple[str, str, Optional[float]]]:
    import csv
    entries: List[Tuple[str, str, Optional[float]]] = []
    with open(path, newline="") as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            hdr = stripped.lstrip("#").strip()
            data_lines.append(hdr)
            continue
        data_lines.append(stripped)

    if not data_lines:
        return entries

    reader = csv.reader(data_lines)
    header = next(reader)
    header_lower = [h.strip().lower() for h in header]

    name_col, smi_col, pk_col = None, None, None
    for i, h in enumerate(header_lower):
        if h in ("name", "compound", "molecule"):
            name_col = i
        elif h in ("smiles", "smi"):
            smi_col = i
        elif h in ("pkhb", "pkbhx", "pk", "pkhbx", "exp_pkbhx", "exp_pkhb"):
            pk_col = i

    if smi_col is None:
        name_col, smi_col, pk_col = 0, 1, 2

    for row in reader:
        if len(row) <= smi_col:
            continue
        nm = row[name_col].strip() if name_col is not None else ""
        smi = row[smi_col].strip()
        if not smi:
            continue
        pk = None
        if pk_col is not None and pk_col < len(row):
            try:
                pk = float(row[pk_col].strip())
            except (ValueError, IndexError):
                pass
        entries.append((nm, smi, pk))

    return entries

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.smiles and not args.file and not args.csv:
        parser.print_help()
        sys.exit(1)

    if args.method is None:
        args.method = "r2scan-3c"

    mol_list: List[Tuple[str, str, Optional[float]]] = []

    if args.csv:
        mol_list.extend(_parse_csv(args.csv))

    if args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    smi = parts[0]
                    nm = parts[1] if len(parts) > 1 else ""
                    mol_list.append((nm, smi, None))

    if args.smiles:
        mol_list.append(("", args.smiles, None))

    if args.json:
        out_path = args.json if args.json.endswith('.jsonl') else args.json + 'l'
        open(out_path, 'w').close()

    t_total = time.time()
    all_results: List[dict] = []
    
    for name, smi, exp_pk in mol_list:
        try:
            result = predict_one(smi, args, name=name, exp_pkbhx=exp_pk)
            all_results.append(result)
            
            if args.json:
                with open(out_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result) + '\n')
                    
        except Exception as e:
            print(f"\n  ERROR on {name or smi}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    dt = time.time() - t_total
    
    if args.json:
        print(f"\n  -> Streamed JSONL output saved to: {out_path}")

    pairs = [(r["mol_pkbhx_pred"], r["mol_pkbhx_exp"])
             for r in all_results
             if r["mol_pkbhx_pred"] is not None
             and r["mol_pkbhx_exp"] is not None]

    print(f"\n{'='*60}")
    print(f"  All done. {len(mol_list)} molecule(s) in {dt:.1f}s")
    if pairs:
        preds = np.array([p for p, _ in pairs])
        exps = np.array([e for _, e in pairs])
        errors = preds - exps
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        r2 = 1.0 - np.sum(errors**2) / np.sum((exps - exps.mean())**2)
        maxe = np.max(np.abs(errors))
        print(f"  Comparison vs. experiment ({len(pairs)} molecules):")
        print(f"    MAE  = {mae:.3f} pKBHX units")
        print(f"    RMSE = {rmse:.3f}")
        print(f"    R²   = {r2:.3f}")
        print(f"    MaxE = {maxe:.3f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
