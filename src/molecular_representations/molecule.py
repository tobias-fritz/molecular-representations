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

@dataclass
class Molecule:
    """A class representing a molecular structure with data from various file formats."""
    
    name: str = "Unnamed"
    
    # Core structure attributes
    atoms: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
        'atom_name': [], 'atom_type': [], 'resname': [], 'resid': [],
        'chain': [], 'segment': [], 'x': [], 'y': [], 'z': [],
        'charge': [], 'mass': [], 'element': [], 'occupancy': [], 
        'beta': [], 'record_name': [],
    }))
    
    # Topology information
    bonds: List[tuple] = field(default_factory=list)
    angles: List[tuple] = field(default_factory=list)
    dihedrals: List[tuple] = field(default_factory=list)
    impropers: List[tuple] = field(default_factory.list)
    
    # Additional properties
    box: Optional[np.ndarray] = None
    properties: Dict = field(default_factory=dict)

    # Covalent radii in Angstroms (from http://alvarez.sites.chemistry.harvard.edu/pdf/JournCompChem_1989_10_2_83.pdf)
    _COVALENT_RADII = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
        'He': 0.28, 'Ne': 0.58, 'Ar': 1.06, 'Kr': 1.21, 'Xe': 1.40,
        'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'Na': 1.66, 'Mg': 1.41,
        'Al': 1.21, 'Si': 1.11, 'K': 2.03, 'Ca': 1.76, 'Ga': 1.22,
        'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Rb': 2.20, 'Sr': 1.95,
        'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
        'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'Au': 1.36, 'Hg': 1.32
    }

    def __init__(self, name: str = "Unnamed", cord: Optional[str] = None, top: Optional[str] = None) -> None:
        self.name = name
        if cord:
            self.read_file(cord)
        if top:
            self.read_file(top)

    def read_file(self, fname: str) -> None:
        """Read atomic data from file"""
        if fname.endswith('.pdb'):
            self._read_pdb(fname)
        elif fname.endswith('.xyz'):
            self._read_xyz(fname)
        elif fname.endswith('.psf'):
            self._read_psf(fname)
        elif fname.endswith('.crd'):
            self._read_crd(fname)
        else:
            raise ValueError(f"Unsupported file format: {fname}")

    def _read_pdb(self, fname: str) -> None:
        """Read atomic data from PDB file"""
        with open(fname, 'r') as ff:
            lines = ff.readlines()
        
        pdb = []
        for line in lines[:]: 
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    pdb.append({'record_name': line[:6].strip(),
                                'serial': int(line[7:12]),
                                'atom_name': line[13:17].strip(),
                                'resname': line[18:21].strip(),
                                'chain': line[22].strip(),
                                'resid': line[23:26].strip(),
                                'x': float(line[31:39].strip()),
                                'y': float(line[39:47].strip()),
                                'z': float(line[47:55].strip()),
                                'occupancy': line[55:61].strip(),
                                'beta': line[61:67].strip(),
                                'segment': line[72:77].strip()})
                except Exception as e:
                    raise Exception(f"Error parsing line: {line.strip()} - {e}")
        
        self.atoms = pd.DataFrame(pdb).set_index('serial')
  
    def _read_xyz(self, fname: str) -> None:
        """Read atomic data from XYZ file"""
        with open(fname, 'r') as ff:
            lines = ff.readlines()
        
        xyz = []
        n_atoms = int(lines[0])
        for line in lines[2:]: 
            try:
                parts = line.split()
                xyz.append({'atom_name': parts[0],
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'z': float(parts[3])})
            except Exception as e:
                raise Exception(f"Error parsing line: {line.strip()} - {e}")
        
        self.atoms = pd.DataFrame(xyz)

    def _read_psf(self, fname: str) -> None:
        """Read atomic data from PSF file"""
        with open(fname, 'r') as ff:
            lines = ff.readlines()
        
        psf = []
        for line in lines[:]: 
            if line.startswith('ATOM'):
                try:
                    parts = line.split()
                    psf.append({'atom_name': parts[2],
                                'resname': parts[3],
                                'resid': int(parts[4]),
                                'charge': float(parts[6]),
                                'mass': float(parts[7]),
                                'segment': parts[8]})
                except Exception as e:
                    raise Exception(f"Error parsing line: {line.strip()} - {e}")
        
        self.atoms = pd.DataFrame(psf)

    def _read_crd(self, fname: str) -> None:
        """Read atomic data from CHARMM CRD file"""
        try:
            with open(fname, 'r') as ff:
                lines = ff.readlines()

            # Skip title line
            n_atoms_line = lines[1].strip()
            is_expanded = "EXT" in n_atoms_line
            
            crd = []
            current_line = 2  # Start after header and atom count

            while current_line < len(lines):
                line = lines[current_line].strip()
                if not line:  # Skip empty lines
                    current_line += 1
                    continue
                    
                try:
                    if is_expanded:
                        # Parse expanded format
                        crd.append({
                            'serial': int(line[0:10]),
                            'residue_number': int(line[10:20]),
                            'resname': line[22:30].strip(),
                            'atom_name': line[32:40].strip(),
                            'x': float(line[40:60]),
                            'y': float(line[60:80]),
                            'z': float(line[80:100]),
                            'segment': line[102:110].strip(),
                            'resid': line[112:120].strip()
                        })
                    else:
                        # Parse normal format
                        crd.append({
                            'serial': int(line[0:5]),
                            'residue_number': int(line[5:10]),
                            'resname': line[11:15].strip(),
                            'atom_name': line[16:20].strip(),
                            'x': float(line[20:30]),
                            'y': float(line[30:40]),
                            'z': float(line[40:50]),
                            'segment': line[51:55].strip(),
                            'resid': line[56:60].strip()
                        })
                except Exception as e:
                    raise ValueError(f"Error parsing line {current_line + 1}: {line}\n{str(e)}")
                
                current_line += 1

            self.atoms = pd.DataFrame(crd).set_index('serial')
            
        except Exception as e:
            raise Exception(f"Error reading CRD file {fname}: {e}")
    
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
            ff.write(f"{len(self.atoms)} !NATOM\n")
            for idx, atom in self.atoms.iterrows():
                ff.write(f"{idx:8} {atom['atom_name']:4} {atom['resname']:4} {atom['resid']:4} {atom['charge']:8.6f} {atom['mass']:8.4f} {atom['segment']:4}\n")
            ff.write("\n")
            ff.write(f"{len(self.bonds)} !NBOND: bonds\n")
            for bond in self.bonds:
                ff.write(f"{bond[0]:8} {bond[1]:8}\n")
            ff.write("\n")
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
        """Return atomic coordinates as numpy array"""
        return self.atoms[['x', 'y', 'z']].values

    def set_coordinates(self, coords: np.ndarray) -> None:
        """Set atomic coordinates from numpy array"""
        if coords.shape[1] != 3:
            raise ValueError("Coordinates must be Nx3 array")
        self.atoms['x'] = coords[:, 0]
        self.atoms['y'] = coords[:, 1]
        self.atoms['z'] = coords[:, 2]

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass"""
        if 'mass' not in self.atoms:
            return self.get_coordinates().mean(axis=0)
        masses = self.atoms['mass'].values
        coords = self.get_coordinates()
        return np.average(coords, weights=masses, axis=0)

    def add_property(self, key: str, value: Any) -> None:
        """Add arbitrary property to the molecule"""
        self.properties[key] = value

    def _get_element(self) -> None:
        """Extract atom element from atom names"""
        raise NotImplementedError

    def compute_bonds(self, element_col: str = 'element', tolerance: float = 1.3) -> List[tuple]:
        """Compute molecular bonds based on atomic distances and covalent radii.
        
        Args:
            element_col: Column name containing element symbols
            tolerance: Factor to multiply sum of covalent radii by
        
        Returns:
            List of tuples containing atom indices of bonded atoms
        """
        # if no elemnt is avialable extract element froma atom_name

        coords = self.get_coordinates()
        elements = self.atoms[element_col].values
        n_atoms = len(coords)
        
        # Compute all pairwise distances
        distances = squareform(pdist(coords))
        
        # Get matrix of covalent radii sums
        radii_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                try:
                    r1 = self._COVALENT_RADII[elements[i]]
                    r2 = self._COVALENT_RADII[elements[j]]
                    radii_matrix[i,j] = radii_matrix[j,i] = (r1 + r2) * tolerance
                except KeyError:
                    # If element not found, use a default value
                    radii_matrix[i,j] = radii_matrix[j,i] = 2.0
        
        # Find bonds where distance is less than allowed maximum
        bonds = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if distances[i,j] <= radii_matrix[i,j]:
                    bonds.append((i+1, j+1))  # +1 for 1-based indexing
        
        self.bonds = bonds
        return bonds

    def compute_angles(self) -> List[tuple]:
        """Compute molecular angles based on existing bonds.
        
        Returns:
            List of tuples (atom1, atom2, atom3, angle) where atom2 is the central atom
            and angle is in degrees
        """
        if not self.bonds:
            # compute bonds
            _ = self.compute_bonds()

        # Create dictionary of bonded atoms for each atom
        bond_dict = {}
        for a1, a2 in self.bonds:
            if a1 not in bond_dict:
                bond_dict[a1] = set()
            if a2 not in bond_dict:
                bond_dict[a2] = set()
            bond_dict[a1].add(a2)
            bond_dict[a2].add(a1)

        # Find all possible angles
        angles = []
        for central_atom in bond_dict:
            # If atom has at least two bonds
            if len(bond_dict[central_atom]) >= 2:
                # Get all possible pairs of bonded atoms
                for atom1, atom2 in combinations(bond_dict[central_atom], 2):
                    # Get coordinates
                    pos1 = self.atoms.loc[atom1, ['x', 'y', 'z']].values
                    pos2 = self.atoms.loc[central_atom, ['x', 'y', 'z']].values
                    pos3 = self.atoms.loc[atom2, ['x', 'y', 'z']].values

                    # Calculate vectors
                    v1 = pos1 - pos2
                    v2 = pos3 - pos2

                    # Calculate angle
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    # Handle numerical errors
                    cos_angle = min(1.0, max(-1.0, cos_angle))
                    angle = np.degrees(np.arccos(cos_angle))

                    angles.append((atom1, central_atom, atom2, angle))

        self.angles = angles
        return angles
    
    def to_pytorch_geometric(self, add_edge_features: bool = True) -> Data:
        """Convert molecular structure to PyTorch Geometric Data object.
        
        Args:
            add_edge_features: Whether to add bond distance features to edges
            
        Returns:
            PyTorch Geometric Data object with:
                - x: Node features (atomic properties)
                - edge_index: Bond connectivity
                - edge_attr: Bond features (distances, if requested)
                - pos: 3D coordinates
        """
        # Ensure bonds are computed
        if not self.bonds:
            self.compute_bonds()
            
        # Create node features
        atom_features = []
        for _, atom in enumerate(self.atoms):
            features = [
                float(atom['mass']) if 'mass' in atom.dtype.names else 0.0,
                float(atom['charge']) if 'charge' in atom.dtype.names else 0.0,
                self._COVALENT_RADII.get(atom['element'], 1.0) if 'element' in atom.dtype.names else 1.0,
            ]
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Create edge index from bonds (0-based indexing)
        edge_index = torch.tensor([[b[0]-1, b[1]-1] for b in self.bonds], dtype=torch.long).t()
        
        # Get positions
        pos = torch.tensor(self.get_coordinates(), dtype=torch.float)
        
        # Compute edge features if requested
        edge_attr = None
        if add_edge_features and edge_index.shape[1] > 0:
            # Compute bond distances
            src, dst = edge_index
            edge_attr = torch.norm(pos[dst] - pos[src], dim=1, keepdim=True)
            
            # Add angle information if available
            if self.angles:
                angle_features = []
                bond_pairs = {(min(b1, b2), max(b1, b2)) for b1, b2 in self.bonds}
                for b1, b2 in zip(src, dst):
                    # Find angles involving this bond
                    bond_angles = [
                        angle[3] for angle in self.angles 
                        if (min(angle[0]-1, angle[2]-1), max(angle[0]-1, angle[2]-1)) == (min(b1, b2), max(b1, b2))
                    ]
                    avg_angle = torch.tensor([sum(bond_angles) / len(bond_angles)]) if bond_angles else torch.tensor([120.0])
                    angle_features.append(avg_angle)
                
                angle_features = torch.stack(angle_features)
                edge_attr = torch.cat([edge_attr, angle_features], dim=1)
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            num_nodes=len(self.atoms)
        )
        
        # Add periodic boundary conditions if available
        if self.box is not None:
            data.box = torch.tensor(self.box, dtype=torch.float)
        
        return data

    def from_pytorch_geometric(self, data: Data) -> None:
        """Initialize molecule from PyTorch Geometric Data object.
        
        Args:
            data: PyTorch Geometric Data object
        """
        num_atoms = data.num_nodes
        self.atoms = AtomArray(num_atoms)
        
        # Set coordinates
        self.atoms.set_coordinates(data.pos.numpy())
        
        # Set bonds (convert to 1-based indexing)
        self.bonds = [(int(src)+1, int(dst)+1) 
                     for src, dst in data.edge_index.t().numpy()]
        
        # Set atomic properties from node features if they match expected format
        if data.x is not None and data.x.shape[1] >= 3:
            self.atoms['mass'] = data.x[:, 0].numpy()
            self.atoms['charge'] = data.x[:, 1].numpy()
        
        # Set box if available
        if hasattr(data, 'box'):
            self.box = data.box.numpy()

    def __repr__(self) -> str:
        return f"Molecule(name='{self.name}', n_atoms={len(self.atoms)})"
