import pytest
import numpy as np
import pandas as pd
from molecular_representations.atom_array import AtomArray

@pytest.fixture
def empty_array():
    """Create an empty AtomArray."""
    return AtomArray(0)

@pytest.fixture
def simple_array():
    """Create a simple AtomArray with 3 atoms."""
    arr = AtomArray(3)
    # Set basic atomic data
    arr['atom_name'] = ['C', 'H', 'O']
    arr['element'] = ['C', 'H', 'O']
    arr['resname'] = ['ALA', 'ALA', 'ALA']
    arr['resid'] = [1, 1, 1]
    arr['x'] = [0.0, 1.0, -1.0]
    arr['y'] = [0.0, 0.0, 0.0]
    arr['z'] = [0.0, 0.0, 0.0]
    return arr

class TestAtomArray:
    def test_initialization(self, empty_array):
        """Test basic initialization."""
        assert empty_array.size == 0
        assert len(empty_array) == 0
        
        # Test with size
        arr = AtomArray(5)
        assert arr.size == 5
        assert len(arr) == 5

    def test_dtype_fields(self, empty_array):
        """Test that all required fields are present."""
        required_fields = {
            'atom_name', 'atom_type', 'resname', 'resid', 'chain',
            'segment', 'x', 'y', 'z', 'charge', 'mass', 'element',
            'occupancy', 'beta', 'record_name'
        }
        actual_fields = set(empty_array.DTYPE.names)
        assert required_fields.issubset(actual_fields)

    def test_coordinate_access(self, simple_array):
        """Test coordinate getter and setter."""
        coords = simple_array.get_coordinates()
        assert coords.shape == (3, 3)
        assert np.allclose(coords[0], [0.0, 0.0, 0.0])
        assert np.allclose(coords[1], [1.0, 0.0, 0.0])
        
        # Test setting coordinates
        new_coords = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ])
        simple_array.set_coordinates(new_coords)
        assert np.allclose(simple_array.get_coordinates(), new_coords)

    def test_item_access(self, simple_array):
        """Test getting and setting items."""
        # Test single field access
        assert simple_array['atom_name'][0] == 'C'
        
        # Test multiple field access
        first_atom = simple_array[0]
        assert first_atom['atom_name'] == 'C'
        assert first_atom['resname'] == 'ALA'
        
        # Test setting values
        simple_array['charge'] = [0.5, -0.5, 0.0]
        assert np.allclose(simple_array['charge'], [0.5, -0.5, 0.0])

    def test_resize(self, simple_array):
        """Test array resizing."""
        # Expand
        simple_array.resize(5)
        assert len(simple_array) == 5
        assert simple_array['atom_name'][0] == 'C'  # Original data preserved
        
        # Shrink
        simple_array.resize(2)
        assert len(simple_array) == 2
        assert simple_array['atom_name'][0] == 'C'  # Original data preserved
        
    def test_pandas_conversion(self, simple_array):
        """Test conversion to/from pandas DataFrame."""
        # To DataFrame
        df = simple_array.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(simple_array)
        assert 'atom_name' in df.columns
        
        # From DataFrame
        new_array = AtomArray.from_dataframe(df)
        assert len(new_array) == len(simple_array)
        assert np.all(new_array['atom_name'] == simple_array['atom_name'])
        assert np.allclose(new_array.get_coordinates(), simple_array.get_coordinates())

    def test_invalid_operations(self, simple_array):
        """Test error handling."""
        # Wrong coordinate shape
        with pytest.raises(ValueError):
            simple_array.set_coordinates(np.zeros((3, 2)))
        
        # Invalid size
        with pytest.raises(ValueError):
            simple_array.set_coordinates(np.zeros((4, 3)))
            
        # Invalid field name
        with pytest.raises(ValueError):
            simple_array['invalid_field'] = [1, 2, 3]

    def test_coordinate_properties(self, simple_array):
        """Test coordinate-specific operations."""
        coords = simple_array.get_coordinates()
        
        # Test coordinate ranges
        assert np.min(coords) == -1.0
        assert np.max(coords) == 1.0
        
        # Test coordinate assignment
        new_coords = np.zeros_like(coords)
        simple_array.set_coordinates(new_coords)
        assert np.allclose(simple_array.get_coordinates(), new_coords)

    def test_data_types(self, simple_array):
        """Test data type handling."""
        # String fields
        assert simple_array['atom_name'].dtype.kind == 'U'
        assert simple_array['resname'].dtype.kind == 'U'
        
        # Numeric fields
        assert simple_array['x'].dtype.kind == 'f'
        assert simple_array['resid'].dtype.kind == 'i'

    def test_array_operations(self, simple_array):
        """Test numpy array operations."""
        # Test array indexing
        subset = simple_array[0:2]
        assert len(subset) == 2
        
        # Test boolean indexing
        mask = simple_array['x'] > 0
        filtered = simple_array[mask]
        assert len(filtered) == 1
        assert filtered['atom_name'][0] == 'H'
