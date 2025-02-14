from typing import List, Tuple
import numpy as np
from scipy.spatial.distance import pdist, squareform

class MoleculeTopology:
    """Topology computation methods for the Molecule class"""
    
    # Covalent radii in Angstroms 
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

    @staticmethod
    def compute_bonds(coords: np.ndarray, elements: np.ndarray, 
                     tolerance: float = 1.3) -> List[tuple]:
        """Compute molecular bonds based on atomic distances and covalent radii."""
        n_atoms = len(coords)
        distances = squareform(pdist(coords))
        
        # Get matrix of covalent radii sums
        radii_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                try:
                    r1 = MoleculeTopology._COVALENT_RADII[elements[i]]
                    r2 = MoleculeTopology._COVALENT_RADII[elements[j]]
                    local_tolerance = tolerance * 1.1 if ('H' in [elements[i], elements[j]]) else tolerance
                    radii_matrix[i,j] = radii_matrix[j,i] = (r1 + r2) * local_tolerance
                except KeyError:
                    radii_matrix[i,j] = radii_matrix[j,i] = 2.0 * tolerance
        
        # Find bonds
        bonds = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if distances[i,j] <= radii_matrix[i,j]:
                    bonds.append((i+1, j+1))  # 1-based indexing
                    
        return bonds

    @staticmethod
    def compute_angles(coords: np.ndarray, bonds: List[tuple]) -> List[tuple]:
        """Compute molecular angles based on existing bonds."""
        # Create bond dictionary
        bond_dict = {}
        for a1, a2 in bonds:
            if a1 not in bond_dict:
                bond_dict[a1] = []
            if a2 not in bond_dict:
                bond_dict[a2] = []
            bond_dict[a1].append(a2)
            bond_dict[a2].append(a1)

        angles = []
        seen = set()

        # Compute angles
        for central in bond_dict:
            if len(bond_dict[central]) < 2:
                continue
                
            for i, atom1 in enumerate(bond_dict[central]):
                for atom2 in bond_dict[central][i+1:]:
                    angle_key = tuple(sorted([atom1, central, atom2]))
                    if angle_key in seen:
                        continue
                    seen.add(angle_key)
                    
                    v1 = coords[atom1-1] - coords[central-1]
                    v2 = coords[atom2-1] - coords[central-1]
                    
                    v1_norm = v1 / np.sqrt(np.sum(v1 * v1))
                    v2_norm = v2 / np.sqrt(np.sum(v2 * v2))
                    
                    dot = np.sum(v1_norm * v2_norm)
                    dot = max(-1.0, min(1.0, dot))
                    angle = np.degrees(np.arccos(dot))
                    
                    angles.append((min(atom1, atom2), central, max(atom1, atom2), angle))

        angles.sort()
        return angles
