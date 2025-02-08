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
def tmp_pdb_file(tmp_path):
    """Create a temporary PDB file for testing."""
    content = """ATOM      1  N   ALA A   1      27.427  24.293   4.635  1.00 23.25      A    N  
ATOM      2  CA  ALA A   1      26.429  25.338   4.709  1.00 23.63      A    C  
ATOM      3  C   ALA A   1      26.917  26.644   4.127  1.00 23.18      A    C  
ATOM      4  O   ALA A   1      27.962  26.686   3.465  1.00 23.94      A    O  
HETATM    5  O   HOH B   1      24.160  24.238   4.211  1.00 38.12      B    O  
END
"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(content)
    return pdb_file

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

class TestPDBReader:
    def test_read_pdb_basic(self, molecule, tmp_pdb_file):
        """Test basic PDB file reading."""
        molecule.read_file(str(tmp_pdb_file))
        
        # Check number of atoms
        assert len(molecule.atoms) == 5
        
        # Check atom names
        expected_names = ['N', 'CA', 'C', 'O', 'O']
        assert [atom['atom_name'] for atom in molecule.atoms] == expected_names
        
        # Check residue names
        expected_resnames = ['ALA', 'ALA', 'ALA', 'ALA', 'HOH']
        assert [atom['resname'] for atom in molecule.atoms] == expected_resnames
        
        # Check chains
        expected_chains = ['A', 'A', 'A', 'A', 'B']
        assert [atom['chain'] for atom in molecule.atoms] == expected_chains
        
        # Check coordinates for first atom
        first_coords = molecule.atoms.get_coordinates()[0]
        expected_coords = np.array([27.427, 24.293, 4.635])
        assert np.allclose(first_coords, expected_coords)

    def test_read_pdb_invalid_format(self, molecule, tmp_path):
        """Test reading invalid PDB file."""
        invalid_file = tmp_path / "invalid.pdb"
        invalid_file.write_text("not a valid pdb file")
        
        with pytest.raises(Exception):
            molecule.read_file(str(invalid_file))

    def test_read_pdb_malformed_coordinates(self, molecule, tmp_path):
        """Test reading PDB with malformed coordinates."""
        content = """ATOM      1  N   ALA A   1      xxx  24.293   4.635  1.00 23.25      A    N  
END"""
        invalid_file = tmp_path / "malformed.pdb"
        invalid_file.write_text(content)
        
        with pytest.raises(Exception):
            molecule.read_file(str(invalid_file))

    def test_read_pdb_hetatm(self, molecule, tmp_path):
        """Test reading PDB with HETATM records."""
        content = """HETATM    1  O   HOH A   1      24.160  24.238   4.211  1.00 38.12      A    O  
END"""
        hetatm_file = tmp_path / "hetatm.pdb"
        hetatm_file.write_text(content)
        
        molecule.read_file(str(hetatm_file))
        assert len(molecule.atoms) == 1
        assert molecule.atoms[0]['record_name'] == 'HETATM'
        assert molecule.atoms[0]['resname'] == 'HOH'

    def test_read_pdb_empty_fields(self, molecule, tmp_path):
        """Test reading PDB with empty optional fields."""
        content = """ATOM      1  N   ALA A   1      27.427  24.293   4.635                          
END"""
        empty_file = tmp_path / "empty_fields.pdb"
        empty_file.write_text(content)
        
        molecule.read_file(str(empty_file))
        assert len(molecule.atoms) == 1
        assert molecule.atoms[0]['occupancy'] == 0.0
        assert molecule.atoms[0]['beta'] == 0.0


