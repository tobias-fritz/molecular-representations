import pytest
import numpy as np
from molecular_representations.molecule import Molecule
from molecular_representations.atom_array import AtomArray

@pytest.fixture
def simple_molecule():
    """Create a simple water molecule."""
    mol = Molecule("water")
    atoms = AtomArray(3)
    
    # Set coordinates for H2O
    coords = [
        [0.0, 0.0, 0.0],  # O
        [0.96, 0.0, 0.0], # H1
        [-0.24, 0.93, 0.0] # H2
    ]
    
    for i, (x, y, z) in enumerate(coords):
        atoms._data[i]['x'] = x
        atoms._data[i]['y'] = y
        atoms._data[i]['z'] = z
    
    # Set elements
    atoms._data[0]['element'] = 'O'
    atoms._data[1]['element'] = 'H'
    atoms._data[2]['element'] = 'H'
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    return mol

@pytest.fixture
def benzene_molecule():
    """Create a benzene molecule."""
    mol = Molecule("benzene")
    atoms = AtomArray(12)  # 6 C and 6 H
    
    # Set coordinates for benzene (planar, regular hexagon)
    radius = 1.40  # C-C bond length
    h_dist = 1.08  # C-H bond length
    
    # Carbon ring
    for i in range(6):
        angle = i * np.pi / 3
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        atoms._data[i]['x'] = x
        atoms._data[i]['y'] = y
        atoms._data[i]['z'] = 0.0
        atoms._data[i]['element'] = 'C'
        
        # Corresponding hydrogen
        h_x = (radius + h_dist) * np.cos(angle)
        h_y = (radius + h_dist) * np.sin(angle)
        atoms._data[i+6]['x'] = h_x
        atoms._data[i+6]['y'] = h_y
        atoms._data[i+6]['z'] = 0.0
        atoms._data[i+6]['element'] = 'H'
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    return mol

def test_simple_water_bonds(simple_molecule):
    """Test bond detection in water molecule."""
    bonds = simple_molecule.compute_bonds()
    
    # Should have 2 bonds: O-H1 and O-H2
    assert len(bonds) == 2
    
    # Convert to set for order-independent comparison
    bond_set = {tuple(sorted(bond)) for bond in bonds}
    expected_bonds = {(1, 2), (1, 3)}  # 1-based indexing
    assert bond_set == expected_bonds

def test_benzene_bonds(benzene_molecule):
    """Test bond detection in benzene molecule."""
    bonds = benzene_molecule.compute_bonds()
    
    # Should have 12 bonds total:
    # 6 C-C bonds in the ring
    # 6 C-H bonds to hydrogens
    assert len(bonds) == 12
    
    # Count C-C and C-H bonds
    c_c_bonds = 0
    c_h_bonds = 0
    
    for b1, b2 in bonds:
        # Convert to 0-based for easier element access
        e1 = benzene_molecule.atoms._data[b1-1]['element']
        e2 = benzene_molecule.atoms._data[b2-1]['element']
        if e1 == 'C' and e2 == 'C':
            c_c_bonds += 1
        elif (e1 == 'C' and e2 == 'H') or (e1 == 'H' and e2 == 'C'):
            c_h_bonds += 1
            
    assert c_c_bonds == 6  # Ring bonds
    assert c_h_bonds == 6  # Bonds to hydrogens

def test_bond_tolerance(simple_molecule):
    """Test effect of tolerance parameter on bond detection."""
    # Very small tolerance - should find no bonds
    no_bonds = simple_molecule.compute_bonds(tolerance=0.5)
    assert len(no_bonds) == 0
    
    # Normal tolerance - should find 2 bonds (O-H bonds)
    normal_bonds = simple_molecule.compute_bonds(tolerance=1.3)
    assert len(normal_bonds) == 2
    
    # Large tolerance - should find 2 bonds (only O-H bonds)
    # H-H distance is still too large even with higher tolerance
    large_bonds = simple_molecule.compute_bonds(tolerance=2.0)
    assert len(large_bonds) == 2

def test_empty_molecule():
    """Test bond computation with empty molecule."""
    mol = Molecule()
    atoms = AtomArray(0)
    mol.atoms = atoms
    mol._coordinates_loaded = True
    
    bonds = mol.compute_bonds()
    assert len(bonds) == 0

def test_single_atom():
    """Test bond computation with single atom."""
    mol = Molecule()
    atoms = AtomArray(1)
    atoms._data[0]['element'] = 'C'
    atoms._data[0]['x'] = 0.0
    atoms._data[0]['y'] = 0.0
    atoms._data[0]['z'] = 0.0
    mol.atoms = atoms
    mol._coordinates_loaded = True
    
    bonds = mol.compute_bonds()
    assert len(bonds) == 0

def test_missing_elements(simple_molecule):
    """Test behavior with missing element information."""
    # Clear element information
    for i in range(len(simple_molecule.atoms)):
        simple_molecule.atoms._data[i]['element'] = ''
        
    bonds = simple_molecule.compute_bonds()
    # Should still find bonds using default radius
    assert len(bonds) > 0

def test_invalid_elements(simple_molecule):
    """Test behavior with invalid element symbols."""
    # Set invalid element symbols
    for i in range(len(simple_molecule.atoms)):
        simple_molecule.atoms._data[i]['element'] = 'XX'
        
    bonds = simple_molecule.compute_bonds()
    # Should still work using default radius
    assert len(bonds) > 0

def test_no_coordinates():
    """Test behavior when coordinates are not loaded."""
    mol = Molecule()
    atoms = AtomArray(3)
    mol.atoms = atoms
    
    with pytest.raises(Exception):
        mol.compute_bonds()

def test_periodic_bonds():
    """Test bond detection with periodic boundary conditions."""
    mol = Molecule()
    atoms = AtomArray(2)
    
    # Place atoms at opposite edges of a 10Å box
    atoms._data[0]['x'] = 0.0
    atoms._data[0]['y'] = 0.0
    atoms._data[0]['z'] = 0.0
    atoms._data[0]['element'] = 'H'
    
    atoms._data[1]['x'] = 9.9
    atoms._data[1]['y'] = 0.0
    atoms._data[1]['z'] = 0.0
    atoms._data[1]['element'] = 'H'
    
    mol.atoms = atoms
    mol._coordinates_loaded = True
    mol.box = np.array([10.0, 10.0, 10.0])

    bonds = mol.compute_bonds()
    # Currently should not find bonds across periodic boundaries
    assert len(bonds) == 0
