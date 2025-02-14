from typing import List, Dict, Optional, Union
import numpy as np
from ..atom_array import AtomArray

class MoleculeIO:
    """IO methods for the Molecule class"""

    @staticmethod
    def merge_coordinates(atoms: AtomArray, new_atoms: AtomArray) -> None:
        """Merge coordinates from new atoms into existing structure."""
        if len(new_atoms) != len(atoms):
            raise Exception(f"Atom count mismatch between topology ({len(atoms)}) and coordinates ({len(new_atoms)})")

        # Store topology fields
        for field in ['charge', 'mass', 'atom_type', 'resname', 'resid', 'segment']:
            if field in atoms.DTYPE.names:
                new_atoms[field] = atoms[field]

        return new_atoms

    @staticmethod
    def read_pdb(fname: str) -> AtomArray:
        """Read atomic data from PDB file"""
        with open(fname, 'r') as ff:
            lines = [line for line in ff.readlines() 
                    if line.startswith(('ATOM', 'HETATM'))]
        
        if not lines:
            raise Exception("No valid ATOM or HETATM records found in PDB file")
            
        atoms = AtomArray(len(lines))
        
        for i, line in enumerate(lines):
            try:
                atoms['record_name'][i] = line[:6].strip()
                atoms['atom_name'][i] = line[12:16].strip()
                atoms['resname'][i] = line[17:20].strip()
                atoms['chain'][i] = line[21:22].strip()
                atoms['resid'][i] = int(line[22:26])
                atoms['x'][i] = float(line[30:38].strip())
                atoms['y'][i] = float(line[38:46].strip())
                atoms['z'][i] = float(line[46:54].strip())
                atoms['occupancy'][i] = float(line[54:60].strip() or "0.0")
                atoms['beta'][i] = float(line[60:66].strip() or "0.0")
                if len(line) >= 77:
                    atoms['segment'][i] = line[72:76].strip()
            except Exception as e:
                raise Exception(f"Error parsing line {i+1}: {line.strip()} - {e}")
                
        return atoms

    @staticmethod
    def read_xyz(fname: str) -> AtomArray:
        """Read atomic data from XYZ file"""
        with open(fname, 'r') as ff:
            lines = [line.strip() for line in ff.readlines() if line.strip()]
            
        if len(lines) < 3:  # Need at least count, comment, and one atom
            raise Exception("Invalid XYZ file format: insufficient lines")
            
        try:
            n_atoms = int(lines[0])
        except ValueError:
            raise Exception("Invalid XYZ file format: first line must be number of atoms")
            
        atom_lines = lines[2:]  # Skip header and comment
        if len(atom_lines) != n_atoms:
            raise Exception(f"XYZ file format error: expected {n_atoms} atoms but found {len(atom_lines)}")
            
        atoms = AtomArray(n_atoms)
        
        for i, line in enumerate(atom_lines):
            try:
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 values per line, got {len(parts)}")
                    
                atoms['atom_name'][i] = parts[0]
                atoms['x'][i] = float(parts[1])
                atoms['y'][i] = float(parts[2])
                atoms['z'][i] = float(parts[3])
                
            except Exception as e:
                raise Exception(f"Error parsing line {i+3}: {line} - {str(e)}")
        
        return atoms

    @staticmethod
    def read_psf(fname: str) -> AtomArray:
        """Read atomic data from PSF file"""
        with open(fname, 'r') as ff:
            lines = [line.strip() for line in ff.readlines()]

        if not lines or 'PSF' not in lines[0]:
            raise Exception("Invalid PSF file format: missing PSF header")

        # Find NATOM section
        natom_idx = None
        n_atoms = 0
        for i, line in enumerate(lines):
            if '!NATOM' in line:
                try:
                    n_atoms = int(line.split('!')[0].strip())
                    natom_idx = i
                    break
                except (ValueError, IndexError):
                    raise Exception("Invalid NATOM section in PSF file")
        
        if natom_idx is None:
            raise Exception("No NATOM section found in PSF file")

        # Get atom lines
        atom_lines = []
        line_idx = natom_idx + 1
        atoms_found = 0
        
        while atoms_found < n_atoms and line_idx < len(lines):
            line = lines[line_idx].strip()
            if line and not line.startswith('!'):
                atom_lines.append(line)
                atoms_found += 1
            line_idx += 1
        
        if len(atom_lines) != n_atoms:
            raise Exception(f"Expected {n_atoms} atoms but found {len(atom_lines)}")
        
        atoms = AtomArray(n_atoms)
        
        for i, line in enumerate(atom_lines):
            try:
                parts = line.split()
                if len(parts) < 8:
                    raise ValueError(f"Missing required fields, got {len(parts)} fields")
                
                atoms['segment'][i] = parts[1]
                atoms['resid'][i] = int(parts[2])
                atoms['resname'][i] = parts[3]
                atoms['atom_name'][i] = parts[4]
                atoms['atom_type'][i] = parts[5]
                atoms['charge'][i] = float(parts[6])
                atoms['mass'][i] = float(parts[7])
                
            except (ValueError, IndexError) as e:
                raise Exception(f"Error parsing line {i+1}: {line} - {str(e)}")
        
        return atoms

    @staticmethod
    def read_crd(fname: str) -> AtomArray:
        """Read atomic data from CHARMM CRD file"""
        try:
            with open(fname, 'r') as ff:
                lines = [line.strip() for line in ff.readlines() if line.strip()]

            if len(lines) < 2:
                raise Exception("Invalid CRD file: insufficient lines")

            try:
                n_atoms = int(lines[1].split()[0])
            except (ValueError, IndexError):
                raise Exception("Invalid atom count in CRD file")

            atoms = AtomArray(n_atoms)
            
            for i, line in enumerate(lines[2:n_atoms+2]):
                try:
                    parts = line.split()
                    if len(parts) < 7:
                        raise ValueError(f"Insufficient fields in line: {line}")
                    
                    atoms['x'][i] = float(parts[4])
                    atoms['y'][i] = float(parts[5])
                    atoms['z'][i] = float(parts[6])
                    
                    atoms['atom_name'][i] = parts[3]
                    atoms['resname'][i] = parts[2]
                    atoms['resid'][i] = int(parts[1])
                    atoms['segment'][i] = parts[7] if len(parts) > 7 else ""
                    
                except (ValueError, IndexError) as e:
                    raise Exception(f"Error parsing line {i+3}: {line} - {str(e)}")

            return atoms

        except Exception as e:
            raise Exception(f"Error reading CRD file {fname}: {str(e)}")

    @staticmethod
    def write_pdb(atoms: AtomArray, fname: str) -> None:
        """Write atomic data to PDB file"""
        with open(fname, 'w') as ff:
            for idx, atom in atoms.iterrows():
                ff.write(f"ATOM  {idx:5} {atom['atom_name']:4} {atom['resname']:3} "
                        f"{atom['chain']:1} {atom['resid']:4}    "
                        f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                        f"{atom['occupancy']:6.2f}{atom['beta']:6.2f}          "
                        f"{atom['segment']:4}\n")

