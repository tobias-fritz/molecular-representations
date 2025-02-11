# Molecular Representations ![Tests](https://github.com/tobias-fritz/molecular-representations/actions/workflows/tests.yml/badge.svg)

A Python library for molecular structure manipulation and analysis. This package provides utilities for working with common molecular file formats, computing molecular properties, and converting between different molecular representations.

## Features

- **File I/O:** Support for multiple molecular file formats
  - PDB (Protein Data Bank)
  - XYZ
  - PSF (Protein Structure File)
  - CRD (CHARMM Coordinate)

- **Structure Analysis**
  - Automatic bond detection 
  - Angle computation 
  - Center of mass calculations
  - Geometric analysis tools

- **Modern Data Structures**
  - Efficient atomic data storage using NumPy arrays
  - PyTorch Geometric integration for machine learning applications
  - Graph-based molecular representations

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from molecular_representations import Molecule

# Load molecule from file
mol = Molecule(cord="example.pdb")

# Compute properties
com = mol.center_of_mass()
mol.compute_bonds()
angles = mol.compute_angles()

# Convert to PyTorch Geometric
graph = mol.to_pytorch_geometric()

# Visualize molecule
mol.draw_molecule(scale_factor=300)
```

## Examples

### 1. Reading Different File Formats

```python
# XYZ file
mol = Molecule(cord="protein.xyz")

# PDB file with topology
mol = Molecule(cord="molecule.pdb", top="molecule.psf")

# CHARMM coordinates
mol = Molecule(cord="structure.crd")
```

### 2. Structure Analysis

```python
# Compute and analyze bonds
mol.compute_bonds(tolerance=1.3)  # Adjust tolerance for bond detection
print(f"Found {len(mol.bonds)} bonds")

# Calculate angles
angles = mol.compute_angles()
print(f"Found {len(angles)} angles")

# Get atom coordinates
coords = mol.get_coordinates()  # Returns Nx3 numpy array
```

### 3. Visualization

```python
# Basic visualization
ax = mol.draw_molecule()

# Customize visualization
ax = mol.draw_molecule(scale_factor=500)  # Larger atoms
plt.show()
```
Example molecular visualization:

![Example molecular visualization](docs/figures/molecule_example.png)img[figure/example_mol.png]

## Test Coverage

Currently, the package has extensive test coverage:
- Core functionality: >50% coverage
- File I/O operations: >50% coverage
- Structure analysis: >95% coverage

Key tested components:
- Bond detection and computation
- Angle calculations
- Coordinate manipulation
- File format parsing
- Data structure operations

## Development

This is a personal utility package that I use for my research in molecular modeling. While it's primarily developed for my own needs, it's designed to be extensible and reusable.

To contribute or modify:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=molecular_representations tests/
```

## License

MIT License

