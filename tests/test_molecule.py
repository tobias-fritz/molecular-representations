import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from molecular_representations.molecule import Molecule
from molecular_representations.atom_array import AtomArray  # Add this import

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
def tmp_psf_file(tmp_path):
    """Create a temporary PSF file for testing."""
    content = """PSF

       1 !NTITLE
 REMARKS PSF CHEM test file

       5 !NATOM
         1 PROT 1    ALA  N    NH3   -0.30000       14.0070           0
         2 PROT 1    ALA  CA   CT1    0.21000       12.0110           0
         3 PROT 1    ALA  CB   CT3   -0.27000       12.0110           0
         4 PROT 1    ALA  C    C      0.51000       12.0110           0
         5 PROT 1    ALA  O    O     -0.51000       15.9990           0

       4 !NBOND: bonds
         1         2         2         3         2         4         4         5

       6 !NTHETA: angles
         1     2     3     1     2     4     3     2     4
         2     4     5

       2 !NPHI: dihedrals
         1     2     4     5     3     2     4     5

END"""
    psf_file = tmp_path / "test.psf"
    psf_file.write_text(content)
    return psf_file

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

class TestPSFReader:
    def test_read_psf_basic(self, molecule, tmp_psf_file):
        """Test basic PSF file reading."""
        molecule.read_file(str(tmp_psf_file))
        
        # Check number of atoms
        assert len(molecule.atoms) == 5
        
        # Check atom properties
        assert list(molecule.atoms['atom_name']) == ['N', 'CA', 'CB', 'C', 'O']
        assert list(molecule.atoms['resname']) == ['ALA'] * 5
        assert list(molecule.atoms['resid']) == [1] * 5
        assert list(molecule.atoms['segment']) == ['PROT'] * 5
        
        # Check charges
        expected_charges = [-0.30, 0.21, -0.27, 0.51, -0.51]
        assert np.allclose(molecule.atoms['charge'], expected_charges)
        
        # Check masses
        expected_masses = [14.007, 12.011, 12.011, 12.011, 15.999]
        assert np.allclose(molecule.atoms['mass'], expected_masses)

    def test_read_psf_invalid_format(self, molecule, tmp_path):
        """Test reading invalid PSF file."""
        invalid_file = tmp_path / "invalid.psf"
        invalid_file.write_text("not a valid psf file")
        
        with pytest.raises(Exception):
            molecule.read_file(str(invalid_file))

    def test_read_psf_missing_fields(self, molecule, tmp_path):
        """Test reading PSF with missing fields."""
        content = """PSF
       1 !NATOM
         1 PROT 1    ALA  N    NH3
END"""
        missing_file = tmp_path / "missing.psf"
        missing_file.write_text(content)
        
        with pytest.raises(Exception):
            molecule.read_file(str(missing_file))

    def test_read_psf_invalid_numbers(self, molecule, tmp_path):
        """Test reading PSF with invalid numeric values."""
        content = """PSF
       1 !NATOM
         1 PROT 1    ALA  N    NH3   xxx           14.0070           0
END"""
        invalid_file = tmp_path / "invalid_numbers.psf"
        invalid_file.write_text(content)
        
        with pytest.raises(Exception):
            molecule.read_file(str(invalid_file))

    def test_read_psf_empty(self, molecule, tmp_path):
        """Test reading empty PSF file."""
        empty_file = tmp_path / "empty.psf"
        empty_file.write_text("")
        
        with pytest.raises(Exception):
            molecule.read_file(str(empty_file))

    def test_read_psf_with_comments(self, molecule, tmp_path):
        """Test reading PSF with comment lines."""
        content = """PSF
! This is a comment
       1 !NATOM
! Another comment
         1 PROT 1    ALA  N    NH3   -0.30000       14.0070           0
END"""
        comment_file = tmp_path / "comments.psf"
        comment_file.write_text(content)
        
        molecule.read_file(str(comment_file))
        assert len(molecule.atoms) == 1

def test_center_of_mass():
    """Test center of mass calculation with and without masses."""
    mol = Molecule()
    
    # Create a simple test molecule with 3 atoms
    atoms = AtomArray(3)
    for i, x in enumerate([0.0, 1.0, 2.0]):
        atoms._data[i]['x'] = x
        atoms._data[i]['y'] = 0.0
        atoms._data[i]['z'] = 0.0
    
    # Test without masses (should be geometric center)
    mol.atoms = atoms
    mol._coordinates_loaded = True
    com = mol.center_of_mass()
    assert np.allclose(com, [1.0, 0.0, 0.0])
    
    # Test with masses
    for i, mass in enumerate([1.0, 2.0, 1.0]):  # Middle atom twice as heavy
        atoms._data[i]['mass'] = mass
    mol.atoms = atoms
    # Expected: (0*1 + 1*2 + 2*1)/(1 + 2 + 1) = 1.0
    com = mol.center_of_mass()
    assert np.allclose(com, [1.0, 0.0, 0.0])
    
    # Test with unequal masses
    for i, mass in enumerate([1.0, 4.0, 1.0]):  # Middle atom 4x as heavy
        atoms._data[i]['mass'] = mass
    mol.atoms = atoms
    # Expected: (0*1 + 1*4 + 2*1)/(1 + 4 + 1) = 1.0
    com = mol.center_of_mass()
    expected = np.array([1.0, 0.0, 0.0])
    assert np.allclose(com, expected)

def test_center_of_mass_empty():
    """Test center of mass calculation with empty molecule."""
    mol = Molecule()
    with pytest.raises(Exception):
        mol.center_of_mass()  # Should raise because no coordinates are loaded

def test_center_of_mass_single_atom():
    """Test center of mass calculation with single atom."""
    mol = Molecule()
    atoms = AtomArray(1)
    atoms._data[0]['x'] = 1.0
    atoms._data[0]['y'] = 2.0
    atoms._data[0]['z'] = 3.0
    atoms._data[0]['mass'] = 1.0
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    com = mol.center_of_mass()
    assert np.allclose(com, [1.0, 2.0, 3.0])


