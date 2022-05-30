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

def XYZ_reader(fname):
    ''' A PSF parser
    param:
        fname :     full or relative file path of the pdb to be read
    '''

    # open the psf file
    with open(fname, 'r') as ff:
        lines = ff.readlines()
    
    # initiate psf file as list
    xyz = []

    #iterate over all lines
    for line in lines[2:]: 

        # read only lines actually containing all information specified in this parser
        if len(line.split()) == 4:
            
            xyz.append({'Element': str(line[:4].strip()),
                        'x': float(line[4:27].strip()),
                        'y': float(line[27:47].strip()),
                        'z': float(line[47:67].strip())})


    # transfer psf to pd Dataframe with atom number as index
    # this conveniently gives the Dict keys as column names 
    xyz = pd.DataFrame(xyz)
    
    return xyz