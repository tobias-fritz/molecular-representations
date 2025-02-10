import numpy as np
from typing import Optional, Dict, List, Union

class AtomArray:
    """Efficient storage of atomic data using structured numpy arrays."""
    
    # Define dtype for structured array
    DTYPE = np.dtype([
        ('atom_name', 'U4'),
        ('atom_type', 'U4'),
        ('resname', 'U4'),
        ('resid', 'i4'),
        ('chain', 'U1'),
        ('segment', 'U4'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('z', 'f8'),
        ('charge', 'f8'),
        ('mass', 'f8'),
        ('element', 'U2'),
        ('occupancy', 'f8'),
        ('beta', 'f8'),
        ('record_name', 'U6')
    ])

    def __init__(self, size: int = 0):
        """Initialize atom array with given size."""
        self._data = np.zeros(size, dtype=self.DTYPE)
        self._size = size

    @property
    def size(self) -> int:
        """Return number of atoms."""
        return self._size

    def resize(self, new_size: int) -> None:
        """Resize the array."""
        new_data = np.zeros(new_size, dtype=self.DTYPE)
        if new_size > self._size:
            new_data[:self._size] = self._data
        else:
            new_data = self._data[:new_size]
        self._data = new_data
        self._size = new_size

    def __getitem__(self, key: Union[int, str, slice]) -> np.ndarray:
        """Get item or field from array."""
        return self._data[key]

    def __setitem__(self, key: Union[int, str, slice], value) -> None:
        """Set item or field in array."""
        self._data[key] = value

    def get_coordinates(self) -> np.ndarray:
        """Return coordinates as Nx3 array."""
        return np.column_stack((self._data['x'], self._data['y'], self._data['z']))

    def set_coordinates(self, coords: np.ndarray) -> None:
        """Set coordinates from Nx3 array.
        
        Args:
            coords: numpy array of shape (N, 3) containing xyz coordinates
            
        Raises:
            ValueError: If coords shape doesn't match (N_atoms, 3)
        """
        if not isinstance(coords, np.ndarray):
            raise ValueError("Coordinates must be a numpy array")
            
        if len(coords.shape) != 2 or coords.shape[1] != 3:
            raise ValueError(f"Coordinates must be Nx3 array, got shape {coords.shape}")
            
        if coords.shape[0] != self._size:
            raise ValueError(f"Number of coordinates ({coords.shape[0]}) must match number of atoms ({self._size})")
            
        self._data['x'] = coords[:, 0]
        self._data['y'] = coords[:, 1]
        self._data['z'] = coords[:, 2]

    @classmethod
    def from_dataframe(cls, df) -> 'AtomArray':
        """Create AtomArray from pandas DataFrame."""
        arr = cls(len(df))
        for field in cls.DTYPE.names:
            if field in df.columns:
                arr[field] = df[field].values
        return arr

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._data)

    def __len__(self) -> int:
        return self._size
