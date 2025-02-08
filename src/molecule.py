from typing import List, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from .atom_array import AtomArray

@dataclass
class Molecule:
    """A class representing a molecular structure with data from various file formats."""
    
    name: str = "Unnamed"
    
    # Core structure attributes
    atoms: AtomArray = field(default_factory=lambda: AtomArray(0))
    
    # Topology information
    bonds: List[tuple] = field(default_factory.list)
    angles: List[tuple] = field(default_factory.list)
    dihedrals: List[tuple] = field(default_factory.list)
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
            lines = [line for line in ff.readlines() 
                    if line.startswith('ATOM') or line.startswith('HETATM')]
        
        # Pre-allocate array
        atoms = AtomArray(len(lines))
        
        for i, line in enumerate(lines):
            try:
                atoms['record_name'][i] = line[:6].strip()
                atoms['atom_name'][i] = line[13:17].strip()
                atoms['resname'][i] = line[18:21].strip()
                atoms['chain'][i] = line[22].strip()
                atoms['resid'][i] = int(line[23:26])
                atoms['x'][i] = float(line[31:39])
                atoms['y'][i] = float(line[39:47])
                atoms['z'][i] = float(line[47:55])
                atoms['occupancy'][i] = float(line[55:61])
                atoms['beta'][i] = float(line[61:67])
                atoms['segment'][i] = line[72:77].strip()
            except Exception as e:
                raise Exception(f"Error parsing line: {line.strip()} - {e}")
        
        self.atoms = atoms

    def _read_xyz(self, fname: str) -> None:
        """Read atomic data from XYZ file"""
        with open(fname, 'r') as ff:
            lines = [line.strip() for line in ff.readlines() if line.strip()]
        
        try:
            n_atoms = int(lines[0])
            atoms = AtomArray(n_atoms)
            
            for i, line in enumerate(lines[2:]):  # Skip header and comment line
                parts = line.split()
                atoms['atom_name'][i] = parts[0]
                atoms['x'][i] = float(parts[1])
                atoms['y'][i] = float(parts[2])
                atoms['z'][i] = float(parts[3])
                
        except Exception as e:
            raise Exception(f"Error parsing XYZ file: {e}")
        
        if len(lines[2:]) != n_atoms:
            raise ValueError(f"Expected {n_atoms} atoms but found {len(lines[2:])}")
            
        self.atoms = atoms

    def _read_psf(self, fname: str) -> None:
        """Read atomic data from PSF file"""
        with open(fname, 'r') as ff:
            lines = [line for line in ff.readlines() if line.startswith('ATOM')]
        
        atoms = AtomArray(len(lines))
        for i, line in enumerate(lines):
            try:
                parts = line.split()
                atoms['atom_name'][i] = parts[2]
                atoms['resname'][i] = parts[3]
                atoms['resid'][i] = int(parts[4])
                atoms['charge'][i] = float(parts[6])
                atoms['mass'][i] = float(parts[7])
                atoms['segment'][i] = parts[8]
            except Exception as e:
                raise Exception(f"Error parsing line: {line.strip()} - {e}")
        
        self.atoms = atoms

    def _read_crd(self, fname: str) -> None:
        """Read atomic data from CHARMM CRD file"""
        try:
            with open(fname, 'r') as ff:
                lines = ff.readlines()

            # Skip title line
            n_atoms_line = lines[1].strip()
            is_expanded = "EXT" in n_atoms_line
            
            # Count valid lines for pre-allocation
            valid_lines = [l for l in lines[2:] if l.strip()]
            atoms = AtomArray(len(valid_lines))
            
            current_idx = 0
            for line in valid_lines:
                try:
                    if is_expanded:
                        # Parse expanded format
                        atoms['serial'][current_idx] = int(line[0:10])
                        atoms['resname'][current_idx] = line[22:30].strip()
                        atoms['atom_name'][current_idx] = line[32:40].strip()
                        atoms['x'][current_idx] = float(line[40:60])
                        atoms['y'][current_idx] = float(line[60:80])
                        atoms['z'][current_idx] = float(line[80:100])
                        atoms['segment'][current_idx] = line[102:110].strip()
                        atoms['resid'][current_idx] = line[112:120].strip()
                    else:
                        # Parse normal format
                        atoms['serial'][current_idx] = int(line[0:5])
                        atoms['resname'][current_idx] = line[11:15].strip()
                        atoms['atom_name'][current_idx] = line[16:20].strip()
                        atoms['x'][current_idx] = float(line[20:30])
                        atoms['y'][current_idx] = float(line[30:40])
                        atoms['z'][current_idx] = float(line[40:50])
                        atoms['segment'][current_idx] = line[51:55].strip()
                        atoms['resid'][current_idx] = line[56:60].strip()
                except Exception as e:
                    raise ValueError(f"Error parsing line {current_idx + 1}: {line}\n{str(e)}")
                
                current_idx += 1

            self.atoms = atoms
            
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
    
    def _write_gro(self, fname: str) -> None:
        """Write atomic data to GRO file"""
        raise NotImplementedError
    
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
        return self.atoms.get_coordinates()

    def set_coordinates(self, coords: np.ndarray) -> None:
        """Set atomic coordinates from numpy array"""
        self.atoms.set_coordinates(coords)

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

    def compute_pairwise_distances(self, kind: str = 'atoms', inverse: bool = False, 
                                 unit: bool = False, normalize: bool = False, 
                                 exp: Optional[float] = None) -> np.ndarray:
        """Compute pairwise distances between atoms or residues.
        
        Args:
            kind: Type of distance calculation ('atoms' or 'residues')
            inverse: Compute inverse of distances (default: False)
            unit: Normalize columns to unit vectors (default: False)
            normalize: Mean normalize matrix (default: False)
            exp: Exponent for inverse distance calculation (default: None)
        
        Returns:
            numpy array of pairwise distances
            
        Raises:
            ValueError: If kind is not supported
            NotImplementedError: For residue distances
        """
        if kind not in ['atoms', 'residues']:
            raise ValueError("kind must be either 'atoms' or 'residues'")
            
        if kind == 'residues':
            raise NotImplementedError("Residue-level distances not yet implemented")
            
        # Get coordinates and compute distances
        coords = self.get_coordinates()
        distances = pd.DataFrame(squareform(pdist(coords)))
        
        # Compute inverse if requested
        if inverse:
            with np.errstate(divide='ignore'):
                distances = 1 / distances
            distances.replace([np.inf], 0, inplace=True)
            if exp is not None:
                distances = distances ** exp
        
        # Normalize to unit vectors if requested
        if unit:
            distances = distances / np.sqrt((distances**2).sum())
            
        # Mean normalize if requested
        if normalize:
            distances = (distances - distances.mean()) / distances.std()
            
        return distances.values

    # Update compute_bonds to use new parameters
    def compute_bonds(self, element_col: str = 'element', tolerance: float = 1.3) -> List[tuple]:
        """Compute molecular bonds based on atomic distances and covalent radii."""
        # Parameter validation
        if not isinstance(element_col, str):
            raise TypeError("element_col must be a string")
        if not isinstance(tolerance, (int, float)):
            raise TypeError("tolerance must be a number")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if element_col not in self.atoms.columns:
            if element_col == 'element':
                self._get_element()  # Try to extract elements from atom names
            else:
                raise ValueError(f"Column {element_col} not found in atoms DataFrame")

        coords = self.get_coordinates()
        elements = self.atoms[element_col].values
        n_atoms = len(coords)
        
        # Use new pairwise distance function
        distances = self.compute_pairwise_distances(kind='atoms')
        
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
    
    def __repr__(self) -> str:
        return f"Molecule(name='{self.name}', n_atoms={len(self.atoms)})"


