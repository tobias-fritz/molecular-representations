#!/usr/bin/env python3
#===============================================================================================
# Script for reading a psf file
# Date: 5.04.2022
# Author: Tobias Fritz based on a script by M. Poeverlein
# Summary:
# Reading XPLOR psf into a pandas df.
#===============================================================================================

import pandas as pd

#===============================================================================================

def PSF_reader(fname):
    ''' A PSF parser
    param:
        fname :     full or relative file path of the pdb to be read
    '''

    # open the psf file
    with open(fname, 'r') as ff:
        lines = ff.readlines()
    
    # initiate psf file as list
    psf = []

    #iterate over all lines
    for line in lines[:]: 

        # read only lines actually containing all information specified in this parser
        if len(line.split()) == 9 and line.split()[0].isdigit() == True and line.split()[1].isdigit() == False:
            psf.append({'Segment': line[11:15].strip(),
                          'Serial': int(line[:11]),
                          'Resname': line[29:36].strip(),
                          'AtomName': line[38:44].strip(),
                          'Atomtype': line[44:52].strip(),
                          'Charge': line[55:71].strip(),
                          'Mass': line[71:80].strip()})


    # transfer psf to pd Dataframe with atom number as index
    # this conveniently gives the Dict keys as column names 
    psf = pd.DataFrame(psf).set_index('Serial')
    
    return psf

