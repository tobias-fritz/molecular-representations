import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from molecular_representations.molecule import Molecule

@pytest.fixture
def tmp_xyz_file(tmp_path):
    """Create a temporary XYZ file for testing."""
    content = """3
test molecule
C    0.000   0.000   0.000
H    1.000   0.000   0.000
H   -1.000   0.000   0.000
"""
    xyz_file = tmp_path / "test.xyz"
    xyz_file.write_text(content)
    return xyz_file

@pytest.fixture
def molecule():
    """Create a clean molecule instance."""
    return Molecule(name="test")

class TestXYZReader:
    def test_read_xyz_basic(self, molecule, tmp_xyz_file):
        """Test basic XYZ file reading."""
        molecule.read_file(str(tmp_xyz_file))
        
        # Check basic properties
        assert len(molecule.atoms) == 3
        assert list(molecule.atoms['atom_name']) == ['C', 'H', 'H']
        
        # Check coordinates
        expected_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0]
        ])
        assert np.allclose(molecule.get_coordinates(), expected_coords)

    def test_read_xyz_invalid_format(self, molecule, tmp_path):
        """Test reading invalid XYZ file."""
        invalid_file = tmp_path / "invalid.xyz"
        invalid_file.write_text("not a valid xyz file")
        
        with pytest.raises(Exception):
            molecule.read_file(str(invalid_file))

    def test_read_xyz_missing_coordinates(self, molecule, tmp_path):
        """Test reading XYZ file with missing coordinates."""
        content = """3
incomplete molecule
C    0.000   0.000
H    1.000   0.000   0.000
H   -1.000   0.000   0.000
"""
        invalid_file = tmp_path / "incomplete.xyz"
        invalid_file.write_text(content)
        
        with pytest.raises(Exception):
            molecule.read_file(str(invalid_file))

    def test_read_xyz_wrong_atom_count(self, molecule, tmp_path):
        """Test reading XYZ file with incorrect atom count."""
        content = """2
wrong count
C    0.000   0.000   0.000
H    1.000   0.000   0.000
H   -1.000   0.000   0.000
"""
        invalid_file = tmp_path / "wrong_count.xyz"
        invalid_file.write_text(content)
        
        with pytest.raises(Exception):
            molecule.read_file(str(invalid_file))

    def test_read_xyz_with_empty_lines(self, molecule, tmp_path):
        """Test reading XYZ file with empty lines."""
        content = """3
extra empty lines

C    0.000   0.000   0.000

H    1.000   0.000   0.000

H   -1.000   0.000   0.000

"""
        xyz_file = tmp_path / "empty_lines.xyz"
        xyz_file.write_text(content)
        
        molecule.read_file(str(xyz_file))
        
        # Check if empty lines were properly skipped
        assert len(molecule.atoms) == 3
        assert list(molecule.atoms['atom_name']) == ['C', 'H', 'H']
        
        # Verify coordinates
        expected_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0]
        ])
        assert np.allclose(molecule.get_coordinates(), expected_coords)


