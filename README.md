# pkbhx Predictor

Site-specific hydrogen-bond basicity (pKBHX) predictor based on the Rowan/Kenny Vmin→pKBHX workflow.

This tool predicts pKBHX by calculating the minimum electrostatic potential (Vmin) near hydrogen bond acceptor atoms and applying functionally-specific linear regressions to compute site-specific pKBHX values.

## Features

- **Quantum Mechanics ESP**: Evaluates electrostatic potential via **Psi4** (r2SCAN-3c).
- **Conformer Generation**: Multi-conformer search via **Auto3D** (AIMNet2) with RDKit ETF/MMFF fallback.
- **Steric Occlusion Correction**: Adjusts basicity predictions for sterically hindered sites (especially amines) based on VdW sphere blockage.
- **Geometry-Informed Lone-Pair Seeding**: Smart search seeding derived from atomic hybridization (e.g. directing search towards specific lone pair projection angles).
- **Caching**: Checkpoint creation to reuse geometries and wavefunctions across runs.
- **Multiple Output Formats**: Generates annotated 2D SVGs, 3D PDBs with Vmin points, ESP cube files, and comprehensive JSON/JSONL outputs.

## Installation

The prediction workflow requires RDKit, Psi4, and Auto3D/PyTorch. Since `psi4` and other chemistry binaries are only available via conda-forge (not PyPI), you cannot use purely pip/uv. However, you can use **Pixi** for blazing-fast installations, or stick to traditional Conda/Mamba.

### Option 1: Fast Installation (Recommended via Pixi)

[Pixi](https://pixi.sh) is a lightning-fast package manager written in Rust that handles both conda-forge and PyPI dependencies concurrently.

1. Install Pixi (if you haven't already):
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
2. Navigate to the codebase and run it simply using pixi. It will automatically resolve and install the blazing-fast environment defined in `pixi.toml`.
```bash
cd pkbhx/
pixi run pkbhx-predict "c1ccncc1"
```

### Option 2: Traditional (Conda / Mamba)

1. Ensure you have [Miniconda](https://docs.anaconda.com/free/miniconda/) or [Mamba](https://github.com/conda-forge/miniforge) installed.
2. Clone this repository and navigate into it.
3. Create the environment from the provided `environment.yml` and explicitly use `uv` for python dependencies to speed it up:

```bash
conda env create -f environment.yml
conda activate pkbhx_prod
```
4. Install the package in editable mode using `uv` for maximum speed:
```bash
pip install uv
uv pip install -e .
```

## Usage

The primary entry point is the `pkbhx-predict` command (or executing `main.py`).

### Basic Prediction
Calculate pKBHX for a single SMILES string. This will output predictions to the console, and generate a 2D SVG rendering (`*_pkbhx.svg`) and a PDB (`*_vmin.pdb`).

```bash
pkbhx-predict "c1ccncc1"
```

### Batch Processing
Run predictions over a file containing multiple SMILES or a CSV file.

```bash
# SMILES file (one per line, can have a name/identifier separated by spaces)
pkbhx-predict -f molecules.smi --json results.jsonl

# CSV file (requires columns Name, SMILES, pKHB)
pkbhx-predict --csv dataset.csv --json results.jsonl
```

### Options

- `--conformers N`: Generate and evaluate the `N` lowest-energy conformers before performing the prediction (default: `1`).
- `--no-steric`: Disable the steric occlusion correction.
- `--esp-cube`: Generate a Gaussian cube file of the electrostatic potential (for visualization in tools like PyMOL or VMD).
- `--cache-dir PATH`: Directory to store/load checkpointed geometries and wavefunctions.

*Note: For a full list of commands, run `pkbhx-predict --help`.*

## References
- Kenny et al., J. Med. Chem. 2016, 59, 4278–4288
- Wagen & Rowan, ChemRxiv 2025, doi:10.26434/chemrxiv-2025-kv6d6-v2
