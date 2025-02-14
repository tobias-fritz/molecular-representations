from typing import List, Optional
import torch
from torch_geometric.data import Data
import numpy as np

class MoleculeML:
    """Machine learning related methods for the Molecule class"""

    @staticmethod
    def to_pytorch_geometric(coords: np.ndarray,
                           bonds: List[tuple],
                           atomic_features: List[List[float]],
                           angles: Optional[List[tuple]] = None,
                           box: Optional[np.ndarray] = None,
                           add_edge_features: bool = True) -> Data:
        """Convert molecular structure to PyTorch Geometric Data object."""
        # Create node features tensor
        x = torch.tensor(atomic_features, dtype=torch.float)
        
        # Create edge index from bonds (0-based indexing)
        edge_index = torch.tensor([[b[0]-1, b[1]-1] for b in bonds], dtype=torch.long).t()
        
        # Get positions tensor
        pos = torch.tensor(coords, dtype=torch.float)
        
        # Compute edge features if requested
        edge_attr = None
        if add_edge_features and edge_index.shape[1] > 0:
            # Compute bond distances
            src, dst = edge_index
            edge_attr = torch.norm(pos[dst] - pos[src], dim=1, keepdim=True)
            
            # Add angle information if available
            if angles:
                angle_features = []
                bond_pairs = {(min(b1, b2), max(b1, b2)) for b1, b2 in bonds}
                for b1, b2 in zip(src, dst):
                    # Find angles involving this bond
                    bond_angles = [
                        angle[3] for angle in angles 
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
            num_nodes=len(coords)
        )
        
        # Add periodic boundary conditions if available
        if box is not None:
            data.box = torch.tensor(box, dtype=torch.float)
        
        return data

    @staticmethod
    def from_pytorch_geometric(data: Data) -> tuple:
        """Convert PyTorch Geometric Data object back to molecular representation.
        
        Returns:
            Tuple containing:
            - coordinates (np.ndarray)
            - bonds (List[tuple])  
            - atomic properties (dict)
            - box vectors (Optional[np.ndarray])
        """
        # Convert coordinates
        coords = data.pos.numpy()
        
        # Convert bonds (to 1-based indexing)
        bonds = [(int(src)+1, int(dst)+1) 
                for src, dst in data.edge_index.t().numpy()]
        
        # Convert atomic properties 
        properties = {}
        if data.x is not None and data.x.shape[1] >= 3:
            properties['mass'] = data.x[:, 0].numpy()
            properties['charge'] = data.x[:, 1].numpy()
            
        # Get box if available
        box = data.box.numpy() if hasattr(data, 'box') else None
        
        return coords, bonds, properties, box
