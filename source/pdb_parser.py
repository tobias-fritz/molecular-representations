#!/usr/bin/env python3
#===============================================================================================
# Script for reading a pdb file
# Date: 5.04.2022
# Author: Tobias Fritz based on a script by M. Poeverlein
# Summary:
# Reading standard pdb specified here (*) into a pandas df.
# (*) https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html 
#===============================================================================================

import pandas as pd

#===============================================================================================

def PDB_reader(fname):
    ''' A PDB file parser
    param:
        fname :     full or relative file path of the pdb to be read
    '''
    try:
        # open the pdb file
        with open(fname, 'r') as ff:
            lines = ff.readlines()
    except Exception as e:
        raise Exception(f"Error reading file {fname}: {e}")
    
    # initiate pdb file as list
    pdb = []

    # iterate over all lines 
    for line in lines[:]: 

        # only consider lines that contain the information
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                pdb.append({'RecordName': line[:6].strip(),
                            'Serial': int(line[7:12]),
                            'AtomName': line[13:17].strip(),
                            'Resname': line[18:21].strip(),
                            'Chain': line[22].strip(),
                            'Resid': line[23:26].strip(),
                            'x': float(line[31:39].strip()),
                            'y': float(line[39:47].strip()),
                            'z': float(line[47:55].strip()),
                            'Occ': line[55:61].strip(),
                            'Beta': line[61:67].strip(),
                            'Segment': line[72:77].strip()})
            except Exception as e:
                raise Exception(f"Error parsing line: {line.strip()} - {e}")

    # transfer pdb to pd Dataframe with atom number as index
    # this conveniently gives the Dict keys as column names 
    pdb = pd.DataFrame(pdb).set_index('Serial')

    return pdb
