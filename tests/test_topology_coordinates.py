import pytest
import numpy as np
from pathlib import Path
from molecular_representations.molecule import Molecule

@pytest.fixture
def tmp_psf_file(tmp_path):
    """Create a temporary PSF topology file."""
    content = """PSF
       5 !NATOM
         1 PROT 1    ALA  N    NH3   -0.30000       14.0070
         2 PROT 1    ALA  CA   CT1    0.21000       12.0110
         3 PROT 1    ALA  CB   CT3   -0.27000       12.0110
         4 PROT 1    ALA  C    C      0.51000       12.0110
         5 PROT 1    ALA  O    O     -0.51000       15.9990
"""
    psf_file = tmp_path / "test.psf"
    psf_file.write_text(content)
    return psf_file

@pytest.fixture
def tmp_pdb_file(tmp_path):
    """Create a temporary PDB coordinate file."""
    content = """ATOM      1  N   ALA A   1      27.427  24.293   4.635  1.00 23.25
ATOM      2  CA  ALA A   1      26.429  25.338   4.709  1.00 23.63
ATOM      3  CB  ALA A   1      25.917  26.644   4.127  1.00 23.18
ATOM      4  C   ALA A   1      26.917  26.686   3.465  1.00 23.94
ATOM      5  O   ALA A   1      24.160  24.238   4.211  1.00 38.12
END"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(content)
    return pdb_file

@pytest.fixture
def tmp_mismatched_pdb(tmp_path):
    """Create a PDB file with different atom names."""
    content = """ATOM      1  N   GLY A   1      27.427  24.293   4.635  1.00 23.25
ATOM      2  CA  GLY A   1      26.429  25.338   4.709  1.00 23.63
ATOM      3  C   GLY A   1      25.917  26.644   4.127  1.00 23.18
END"""
    pdb_file = tmp_path / "mismatched.pdb"
    pdb_file.write_text(content)
    return pdb_file

@pytest.fixture
def tmp_crd_file(tmp_path):
    """Create a temporary CRD coordinate file."""
    content = """* CHARMM coordinates
    5
    1    1 ALA  N      27.427  24.293   4.635 PROT 1
    2    1 ALA  CA     26.429  25.338   4.709 PROT 1
    3    1 ALA  CB     25.917  26.644   4.127 PROT 1
    4    1 ALA  C      26.917  26.686   3.465 PROT 1
    5    1 ALA  O      24.160  24.238   4.211 PROT 1"""
    crd_file = tmp_path / "test.crd"
    crd_file.write_text(content)
    return crd_file

class TestTopologyCoordinates:
    def test_psf_pdb_matching(self, tmp_psf_file, tmp_pdb_file):
        """Test reading matching PSF topology and PDB coordinates."""
        mol = Molecule()
        mol.read_file(str(tmp_psf_file))  # Read topology first
        mol.read_file(str(tmp_pdb_file))  # Then coordinates
        
        # Check if topology info was preserved
        assert len(mol.atoms) == 5
        assert all(mol.atoms['atom_name'] == ['N', 'CA', 'CB', 'C', 'O'])
        assert all(mol.atoms['charge'] == [-0.3, 0.21, -0.27, 0.51, -0.51])
        
        # Check if coordinates were updated
        coords = mol.get_coordinates()
        assert coords.shape == (5, 3)
        assert np.allclose(coords[0], [27.427, 24.293, 4.635])

    def test_psf_crd_matching(self, tmp_psf_file, tmp_crd_file):
        """Test reading matching PSF topology and CRD coordinates."""
        mol = Molecule()
        mol.read_file(str(tmp_psf_file))
        mol.read_file(str(tmp_crd_file))
        
        assert len(mol.atoms) == 5
        coords = mol.get_coordinates()
        assert coords.shape == (5, 3)
        assert np.allclose(coords[0], [27.427, 24.293, 4.635])

    def test_mismatched_topology_coords(self, tmp_psf_file, tmp_mismatched_pdb):
        """Test reading mismatched topology and coordinate files."""
        mol = Molecule()
        mol.read_file(str(tmp_psf_file))
        
        with pytest.raises(Exception) as exc_info:
            mol.read_file(str(tmp_mismatched_pdb))
        assert "Atom count mismatch" in str(exc_info.value)

    def test_wrong_order_reading(self, tmp_psf_file, tmp_pdb_file):
        """Test reading coordinates before topology."""
        mol = Molecule()
        mol.read_file(str(tmp_pdb_file))  # Coordinates first
        mol.read_file(str(tmp_psf_file))  # Then topology
        
        # Check if both coordinate and topology information is present
        assert len(mol.atoms) == 5
        assert 'charge' in mol.atoms.DTYPE.names
        assert np.all(mol.get_coordinates() != 0)

    def test_multiple_coordinate_updates(self, tmp_psf_file, tmp_pdb_file, tmp_crd_file):
        """Test updating coordinates multiple times."""
        mol = Molecule()
        mol.read_file(str(tmp_psf_file))
        mol.read_file(str(tmp_pdb_file))
        
        # Store PDB coordinates
        pdb_coords = mol.get_coordinates().copy()
        
        # Update with CRD coordinates
        mol.read_file(str(tmp_crd_file))
        crd_coords = mol.get_coordinates()
        
        # Coordinates should match between PDB and CRD
        assert np.allclose(pdb_coords, crd_coords)

    def test_invalid_file_combinations(self, tmp_psf_file, tmp_path):
        """Test invalid file format combinations."""
        mol = Molecule()
        mol.read_file(str(tmp_psf_file))
        
        # Create invalid coordinate file
        invalid_file = tmp_path / "invalid.xyz"
        invalid_file.write_text("Invalid content")
        
        with pytest.raises(Exception):
            mol.read_file(str(invalid_file))

    def test_constructor_topology_coords(self, tmp_psf_file, tmp_pdb_file):
        """Test providing topology and coordinates in constructor."""
        mol = Molecule(
            name="test",
            cord=str(tmp_pdb_file),
            top=str(tmp_psf_file)
        )
        
        assert len(mol.atoms) == 5
        assert 'charge' in mol.atoms.DTYPE.names
        assert np.all(mol.get_coordinates() != 0)

    def test_incomplete_topology(self, tmp_path, tmp_pdb_file):
        """Test reading incomplete topology file."""
        # Create incomplete PSF
        incomplete_psf = tmp_path / "incomplete.psf"
        incomplete_psf.write_text("PSF\n  3 !NATOM\n")
        
        mol = Molecule()
        with pytest.raises(Exception):
            mol.read_file(str(incomplete_psf))
