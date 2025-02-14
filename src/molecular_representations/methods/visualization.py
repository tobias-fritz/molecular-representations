import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

class MoleculeVisualization:
    """Visualization methods for the Molecule class"""

    @staticmethod
    def draw_molecule(coords: np.ndarray, 
                     elements: np.ndarray,
                     bonds: List[tuple], 
                     element_colors: Dict[str, str],
                     covalent_radii: Dict[str, float],
                     scale_factor: float = 300) -> plt.Axes:
        """Draw the molecule in 3D using matplotlib.
        
        Args:
            coords: Nx3 array of atomic coordinates
            elements: Array of element symbols
            bonds: List of (atom1, atom2) tuples representing bonds
            element_colors: Dictionary mapping elements to colors
            covalent_radii: Dictionary mapping elements to radii
            scale_factor: Factor to scale atom sizes by
            
        Returns:
            matplotlib Axes object with the 3D plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot atoms
        for i, element in enumerate(elements):
            color = element_colors.get(element, 'grey')
            size = covalent_radii.get(element, 1.0) * scale_factor
            ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                      color=color, s=size)

        # Plot bonds
        for bond in bonds:
            start = coords[bond[0]-1]  # Convert to 0-based indexing
            end = coords[bond[1]-1]
            ax.plot([start[0], end[0]], 
                   [start[1], end[1]], 
                   [start[2], end[2]], 
                   color='black')

        # Set equal aspect ratio and viewing angle
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=45)  # Adjust default viewing angle
        
        # Remove axis labels and ticks for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        return ax
