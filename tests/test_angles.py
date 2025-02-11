import pytest
import numpy as np
from molecular_representations.molecule import Molecule
from molecular_representations.atom_array import AtomArray

@pytest.fixture
def water_molecule():
    """Create a water molecule with known angle."""
    mol = Molecule("water")
    atoms = AtomArray(3)
    
    # Use exact coordinates for water with 104.5° angle
    # H-O bond length = 0.96 Å
    atoms._data[0]['x'] = 0.0   # O
    atoms._data[0]['y'] = 0.0
    atoms._data[0]['z'] = 0.0
    atoms._data[0]['element'] = 'O'

    atoms._data[1]['x'] = 0.96  # H1
    atoms._data[1]['y'] = 0.0
    atoms._data[1]['z'] = 0.0
    atoms._data[1]['element'] = 'H'

    atoms._data[2]['x'] = -0.25  # H2
    atoms._data[2]['y'] = 0.93
    atoms._data[2]['z'] = 0.0
    atoms._data[2]['element'] = 'H'
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    return mol

@pytest.fixture
def methane_molecule():
    """Create a methane molecule with tetrahedral angles."""
    mol = Molecule("methane")
    atoms = AtomArray(5)
    
    # C at origin
    atoms._data[0]['x'] = 0.0
    atoms._data[0]['y'] = 0.0
    atoms._data[0]['z'] = 0.0
    atoms._data[0]['element'] = 'C'
    
    ch_bond = 1.09  # typical C-H bond length
    # Exact tetrahedral coordinates
    h_coords = [
        [0, 0, ch_bond],  # H1 directly above
        [ch_bond * 0.9428, 0, -ch_bond * 0.3333],  # H2
        [-ch_bond * 0.4714, ch_bond * 0.8165, -ch_bond * 0.3333],  # H3
        [-ch_bond * 0.4714, -ch_bond * 0.8165, -ch_bond * 0.3333]  # H4
    ]
    
    for i, (x, y, z) in enumerate(h_coords):
        atoms._data[i+1]['x'] = x
        atoms._data[i+1]['y'] = y
        atoms._data[i+1]['z'] = z
        atoms._data[i+1]['element'] = 'H'
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    return mol

def test_water_angle(water_molecule):
    """Test angle calculation for water molecule."""
    angles = water_molecule.compute_angles()

    # Should have one H-O-H angle
    assert len(angles) == 1
    
    # Check angle value (104.5° for water)
    _, central, _, angle = angles[0]
    assert central == 1  # O should be central atom
    assert np.isclose(angle, 104.5, atol=1.0)  # Check with tighter tolerance

def test_methane_angles(methane_molecule):
    """Test angle calculations for methane."""
    angles = methane_molecule.compute_angles()
    
    # Should have 6 H-C-H angles (4 choose 2 = 6)
    assert len(angles) == 6
    
    # All angles should be ~109.5° (tetrahedral)
    for _, central, _, angle in angles:
        assert central == 1  # C should be central atom
        assert np.isclose(angle, 109.5, atol=1.0)

def test_linear_molecule():
    """Test angle calculation for a linear molecule (CO2)."""
    mol = Molecule("CO2")
    atoms = AtomArray(3)
    
    # Set up linear CO2
    coords = [
        [-1.0, 0.0, 0.0],  # O
        [0.0, 0.0, 0.0],   # C
        [1.0, 0.0, 0.0]    # O
    ]
    
    for i, (x, y, z) in enumerate(coords):
        atoms._data[i]['x'] = x
        atoms._data[i]['y'] = y
        atoms._data[i]['z'] = z
        atoms._data[i]['element'] = ['O', 'C', 'O'][i]
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    
    angles = mol.compute_angles()
    assert len(angles) == 1
    _, _, _, angle = angles[0]
    assert np.isclose(angle, 180.0, atol=0.1)

def test_no_angles():
    """Test molecules that should have no angles."""
    # Single atom
    mol = Molecule()
    atoms = AtomArray(1)
    atoms._data[0]['x'] = 0.0
    atoms._data[0]['y'] = 0.0
    atoms._data[0]['z'] = 0.0
    atoms._data[0]['element'] = 'H'
    mol.atoms = atoms
    mol._coordinates_loaded = True
    
    angles = mol.compute_angles()
    assert len(angles) == 0
    
    # Diatomic molecule (no angles possible)
    atoms = AtomArray(2)
    atoms._data[0]['element'] = 'H'
    atoms._data[1]['element'] = 'H'
    atoms._data[0]['x'] = 0.0
    atoms._data[1]['x'] = 1.0
    mol.atoms = atoms
    
    angles = mol.compute_angles()
    assert len(angles) == 0

def test_angle_without_bonds():
    """Test angle computation triggers bond computation."""
    mol = Molecule("water")
    atoms = AtomArray(3)
    
    # Set up water molecule
    for i in range(3):
        atoms._data[i]['element'] = ['O', 'H', 'H'][i]
    atoms._data[0]['x'] = 0.0
    atoms._data[1]['x'] = 1.0
    atoms._data[2]['x'] = -1.0
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    
    # Should compute bonds first, then angles
    angles = mol.compute_angles()
    assert len(angles) > 0
    assert len(mol.bonds) > 0

def test_numerical_stability():
    """Test angle calculation with nearly linear/degenerate cases."""
    mol = Molecule()
    atoms = AtomArray(3)
    
    # Almost linear arrangement (179.9°)
    atoms._data[0]['x'] = 0.0
    atoms._data[1]['x'] = 1.0
    atoms._data[2]['x'] = 2.0
    atoms._data[2]['y'] = 0.001  # Slightly off axis
    
    for i in range(3):
        atoms._data[i]['element'] = 'C'
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    
    angles = mol.compute_angles()
    assert len(angles) == 1
    _, _, _, angle = angles[0]
    assert np.isclose(angle, 179.9, atol=0.1)

def test_angle_ordering():
    """Test that angles are reported with consistent atom ordering."""
    mol = Molecule()
    atoms = AtomArray(3)
    
    # Simple triangle
    coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0]
    ]
    
    for i, (x, y, z) in enumerate(coords):
        atoms._data[i]['x'] = x
        atoms._data[i]['y'] = y
        atoms._data[i]['z'] = z
        atoms._data[i]['element'] = 'C'
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    
    angles = mol.compute_angles()
    
    # Check that for each angle, first atom index < last atom index
    for a1, _, a3, _ in angles:
        assert a1 < a3

def test_multiple_central_atoms():
    """Test molecule with multiple atoms having multiple bonds (ethane)."""
    pass  # Later
    

