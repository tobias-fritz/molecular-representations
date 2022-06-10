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

    colspecs = [(0,6),(6,11),(12,16),(17,21),(21,22),(22,26),
                (30,38),(38,46),(46,54),(54,60),(60,66),(72,76)]
    names = ['RecordName','Serial','AtomName','Resname','Chain',
                'Resid','x','y','z','Occ','Beta','Segment']

    # read pdb file
    pdb = pd.read_fwf(fname,colspecs = colspecs,names=names)

    # remove lines that dont contain our molecular data and set Serial as index
    pdb = pdb.drop(pdb.loc[pdb['RecordName']=="ATOM"].index)
    pdb = pdb.set_index('Serial')

    return pdb
