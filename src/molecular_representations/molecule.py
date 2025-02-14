from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import Optional, Dict, Any, Tuple
from .atom_array import AtomArray
import matplotlib.pyplot as plt
from .methods.io import MoleculeIO
from .methods.topology import MoleculeTopology
from .methods.visualization import MoleculeVisualization
from .methods.ml import MoleculeML

@dataclass
class Molecule:
    """A class representing a molecular structure with data from various file formats."""
    
    name: str = "Unnamed"
    atoms: AtomArray = field(default_factory=lambda: AtomArray(0))
    bonds: List[tuple] = field(default_factory=list)
    angles: List[tuple] = field(default_factory=list)
    box: Optional[np.ndarray] = None
    
    # Constants moved to MoleculeTopology class
    _ELEMENT_COLORS = {
        'H': 'grey', 'C': 'k', 'N': 'b', 'O': 'r',
        'F': 'green', 'P': 'orange', 'S': 'yellow',
        'Cl': 'green', 'Br': 'brown', 'I': 'purple'
    }

    def __init__(self, name: str = "Unnamed", cord: Optional[str] = None, top: Optional[str] = None):
        """Initialize molecule with optional topology and coordinate files.
        
        Args:
            name: Molecule name
            cord: Path to coordinate file
            top: Path to topology file
        """
        self.name = name
        self._coordinates_loaded = False
        self.atoms = AtomArray(0)
        self.bonds = []
        
        # Read topology first if provided
        if top:
            self.read_file(top)
            
        # Then read coordinates if provided
        if cord:
            self.read_file(cord)
    
    def read_file(self, fname: str) -> None:
        """Read atomic data from file."""
        # Store existing state
        existing_atoms = None
        had_coordinates = False
        had_elements = False
        existing_coords = None
        
        if hasattr(self, 'atoms') and len(self.atoms) > 0:
            existing_atoms = self.atoms.copy()
            had_coordinates = self._coordinates_loaded
            had_elements = 'element' in self.atoms.DTYPE.names and any(self.atoms._data['element'])
            if had_coordinates:
                existing_coords = self.get_coordinates()
        
        try:
            # Read new file based on extension
            if fname.endswith('.psf'):  # Topology file
                new_atoms = MoleculeIO.read_psf(fname)
                # Maintain coordinates if they existed
                if existing_coords is not None:
                    new_atoms.set_coordinates(existing_coords)
                    self._coordinates_loaded = True
                else:
                    self._coordinates_loaded = False
                self.atoms = new_atoms
            else:  # Coordinate file
                if fname.endswith('.pdb'):
                    new_atoms = MoleculeIO.read_pdb(fname)
                elif fname.endswith('.xyz'):
                    new_atoms = MoleculeIO.read_xyz(fname)
                elif fname.endswith('.crd'):
                    new_atoms = MoleculeIO.read_crd(fname)
                else:
                    raise ValueError(f"Unsupported file format: {fname}")

                # Merge if we had existing atoms
                if existing_atoms is not None:
                    new_atoms = MoleculeIO.merge_coordinates(existing_atoms, new_atoms)
                    
                self.atoms = new_atoms
                self._coordinates_loaded = True

            # Try to extract elements if needed
            if not had_elements and 'atom_name' in self.atoms.DTYPE.names:
                try:
                    self._get_element()
                except Exception:
                    pass

        except Exception as e:
            if existing_atoms is not None:
                self.atoms = existing_atoms
                self._coordinates_loaded = had_coordinates
            raise e

    def write_file(self, fname: str) -> None:
        """Write atomic data to file."""
        if fname.endswith('.pdb'):
            MoleculeIO.write_pdb(self.atoms, fname)
        # Add other file formats...
            
    def compute_bonds(self, element_col: str = 'element', tolerance: float = 1.3) -> List[tuple]:
        """Compute molecular bonds."""
        if not self._coordinates_loaded:
            raise Exception("No coordinates loaded")
            
        coords = self.get_coordinates()
        elements = self.atoms._data[element_col]
        self.bonds = MoleculeTopology.compute_bonds(coords, elements, tolerance)
        return self.bonds

    def compute_angles(self) -> List[tuple]:
        """Compute molecular angles."""
        if not hasattr(self, 'bonds') or not self.bonds:
            self.compute_bonds()
            
        coords = self.get_coordinates()
        self.angles = MoleculeTopology.compute_angles(coords, self.bonds)
        return self.angles
        
    def draw_molecule(self, scale_factor: float = 300) -> plt.Axes:
        """Draw the molecule in 3D."""
        if not self._coordinates_loaded:
            raise Exception("No coordinates loaded")

        if not self.bonds:
            self.compute_bonds()
            
        # Make sure elements are available
        if 'element' not in self.atoms.DTYPE.names or not any(self.atoms._data['element']):
            self._get_element()
            
        coords = self.get_coordinates()
        elements = self.atoms._data['element']
        
        return MoleculeVisualization.draw_molecule(
            coords=coords,
            elements=elements,
            bonds=self.bonds,
            element_colors=self._ELEMENT_COLORS,
            covalent_radii=MoleculeTopology._COVALENT_RADII,
            scale_factor=scale_factor
        )

    def _merge_coordinates(self, new_atoms: AtomArray) -> None:
        """Merge coordinates from new atoms into existing structure."""
        if len(new_atoms) != len(self.atoms):
            raise Exception(f"Atom count mismatch between topology ({len(self.atoms)}) and coordinates ({len(new_atoms)})")
            
        # Store topology information
        old_atoms = self.atoms.copy()
        
        # Update atoms with new coordinates
        self.atoms = new_atoms
        
        # Copy back topology fields
        for field in ['charge', 'mass', 'atom_type', 'resname', 'resid', 'segment']:
            if field in old_atoms.DTYPE.names:
                self.atoms[field] = old_atoms[field]
                
        self._coordinates_loaded = True

    def _merge_topology(self, old_atoms: AtomArray) -> None:
        """Merge topology information from old atoms."""
        if len(old_atoms) != len(self.atoms):
            raise Exception(f"Atom count mismatch: old={len(old_atoms)}, new={len(self.atoms)}")
            
        # Store current coordinates if they exist
        coords = None
        if self._coordinates_loaded:
            coords = self.atoms.get_coordinates()
            
        # Copy topology fields
        for field in ['charge', 'mass', 'atom_type']:
            if field in old_atoms.DTYPE.names:
                self.atoms[field] = old_atoms[field]
                
        # Restore coordinates if they existed
        if coords is not None:
            self.atoms.set_coordinates(coords)

    def write_file(self, fname: str) -> None:
        """Write atomic data to file"""
        if fname.endswith('.pdb'):
            self._write_pdb(fname)
        elif fname.endswith('.xyz'):
            self._write_xyz(fname)
        elif fname.endswith('.psf'):
            self._write_psf(fname)
        elif fname.endswith('.crd'):
            self._write_crd(fname)
        else:
            raise ValueError(f"Unsupported file format: {fname}")

    def _write_pdb(self, fname: str) -> None:
        """Write atomic data to PDB file"""
        with open(fname, 'w') as ff:
            for idx, atom in self.atoms.iterrows():
                ff.write(f"ATOM  {idx:5} {atom['atom_name']:4} {atom['resname']:3} {atom['chain']:1} {atom['resid']:4}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}{atom['occupancy']:6.2f}{atom['beta']:6.2f}          {atom['segment']:4}\n")
      
    def _write_xyz(self, fname: str) -> None:
        """Write atomic data to XYZ file"""
        with open(fname, 'w') as ff:
            ff.write(f"{len(self.atoms)}\n")
            for idx, atom in self.atoms.iterrows():
                ff.write(f"{atom['atom_name']} {atom['x']} {atom['y']} {atom['z']}\n")
    
    def _write_psf(self, fname: str) -> None:
        """Write atomic data to PSF file"""
        with open(fname, 'w') as ff:
            ff.write(f"PSF\n\n")
            ff.write(f"{len(self.angles)} !NTHETA: angles\n")
            for angle in self.angles:
                ff.write(f"{angle[0]:8} {angle[1]:8} {angle[2]:8}\n")
            ff.write("\n")
            ff.write(f"{len(self.dihedrals)} !NPHI: dihedrals\n")
            for dihedral in self.dihedrals:
                ff.write(f"{dihedral[0]:8} {dihedral[1]:8} {dihedral[2]:8} {dihedral[3]:8}\n")
            ff.write("\n")
            ff.write(f"{len(self.impropers)} !NIMPHI: impropers\n")
            for improper in self.impropers:
                ff.write(f"{improper[0]:8} {improper[1]:8} {improper[2]:8} {improper[3]:8}\n")

    def _write_crd(self, fname: str, format: str = "normal") -> None:
        """Write atomic data to CHARMM coordinate file
        
        Args:
            fname: Output filename
            format: Format type ('normal' or 'expanded')
        """
        # Validate and potentially upgrade format
        format = format.lower()
        if format not in ["normal", "expanded"]:
            raise ValueError("Format must be 'normal' or 'expanded'")

        # Check if we need to upgrade to expanded format
        if format == "normal":
            if len(self.atoms) >= 10e4:
                print("Using expanded format, number of atoms is greater than 99999")
                format = "expanded"
            if self.atoms['segment'].str.len().max() > 4:
                print("Using expanded format, at least one segment name is more than 4 characters")
                format = "expanded"
            if self.atoms['resname'].str.len().max() > 4:
                print("Using expanded format, at least one residue name is more than 4 characters")
                format = "expanded"

        with open(fname, 'w') as f:
            f.write("* CHARMM coordinates\n")
            if format == "normal":
                f.write(f"{len(self.atoms):5d}\n")
            else:  # expanded
                f.write(f"{len(self.atoms):10d}  EXT\n")

            # Track residue changes for residue counting
            prev_resid = None
            prev_segid = None
            res_counter = 1
            
            # Write coordinates
            for idx, atom in self.atoms.iterrows():
                # Update residue counter on residue or segment change
                if prev_resid is not None:
                    if prev_resid != atom['resid'] or prev_segid != atom['segment']:
                        res_counter += 1
                prev_resid = atom['resid']
                prev_segid = atom['segment']
                
                if format == "normal":
                    f.write(f"{idx:5d}{res_counter:5d} {atom['resname']:<4s} {atom['atom_name']:<4s}"
                           f"{atom['x']:10.5f}{atom['y']:10.5f}{atom['z']:10.5f} "
                           f"{atom['segment']:<4s} {atom['resid']:<4s}{0.0:10.5f}\n")
                else:  # expanded
                    f.write(f"{idx:10d}{res_counter:10d}  {atom['resname']:<8s}  {atom['atom_name']:<8s}"
                           f"{atom['x']:20.10f}{atom['y']:20.10f}{atom['z']:20.10f}  "
                           f"{atom['segment']:<8s}  {atom['resid']:<8s}{0.0:20.10f}\n")

    def get_coordinates(self) -> np.ndarray:
        """Return atomic coordinates as numpy array."""
        if not self._coordinates_loaded:
            raise Exception("No coordinates loaded")
        return self.atoms.get_coordinates()

    def set_coordinates(self, coords: np.ndarray) -> None:
        """Set atomic coordinates from numpy array."""
        self.atoms.set_coordinates(coords)
        self._coordinates_loaded = True

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass"""
        if not self._coordinates_loaded:
            raise Exception("No coordinates loaded")
            
        coords = np.column_stack([
            self.atoms._data['x'],
            self.atoms._data['y'],
            self.atoms._data['z']
        ])
        
        if 'mass' in self.atoms.DTYPE.names and np.any(self.atoms._data['mass'] != 0):
            masses = self.atoms._data['mass']
            return np.average(coords, weights=masses, axis=0)
        
        return coords.mean(axis=0)

    def add_property(self, key: str, value: Any) -> None:
        """Add arbitrary property to the molecule"""
        self.properties[key] = value

    def _get_element(self) -> None:
        """Extract atom elements from atom names.
        
        Populates the 'element' field of atoms by examining atom_name field.
        Uses first character if it's not a number, otherwise second character.
        """
        if 'atom_name' not in self.atoms.DTYPE.names:
            raise Exception("No atom names available to extract elements")
            
        for i, atom in enumerate(self.atoms):
            atom_name = atom['atom_name']
            if not atom_name:
                continue
                
            # Get first character if not numeric, otherwise second character
            element = atom_name[0] if atom_name[0] not in "0123456789" else atom_name[1]
            self.atoms._data['element'][i] = element

    def to_pytorch_geometric(self, add_edge_features: bool = True) -> Data:
        """Convert molecular structure to PyTorch Geometric Data object."""
        if not self.bonds:
            self.compute_bonds()
            
        # Create atomic features list
        atom_features = []
        for _, atom in enumerate(self.atoms):
            features = [
                float(atom['mass']) if 'mass' in atom.dtype.names else 0.0,
                float(atom['charge']) if 'charge' in atom.dtype.names else 0.0,
                self._COVALENT_RADII.get(atom['element'], 1.0) if 'element' in atom.dtype.names else 1.0,
            ]
            atom_features.append(features)
        
        return MoleculeML.to_pytorch_geometric(
            coords=self.get_coordinates(),
            bonds=self.bonds,
            atomic_features=atom_features,
            angles=self.angles if hasattr(self, 'angles') else None,
            box=self.box,
            add_edge_features=add_edge_features
        )

    def from_pytorch_geometric(self, data: Data) -> None:
        """Initialize molecule from PyTorch Geometric Data object."""
        coords, bonds, properties, box = MoleculeML.from_pytorch_geometric(data)
        
        # Set coordinates
        self.atoms.set_coordinates(coords)
        self._coordinates_loaded = True
        
        # Set bonds
        self.bonds = bonds
        
        # Set atomic properties
        for key, values in properties.items():
            self.atoms[key] = values
            
        # Set box if available
        self.box = box

    def __len__(self) -> int:
        """Return the number of atoms in the molecule."""
        return len(self.atoms)

    def __repr__(self) -> str:
        return f"Molecule(name='{self.name}', n_atoms={len(self.atoms)})"
