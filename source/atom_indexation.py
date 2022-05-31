#!/usr/bin/env python3
#===============================================================================================
# Script for scoring the atom indexing in xyz files of two at the very least isomeric structures
# Date: 30.05.2022
# Author: Tobias Fritz
# Summary:
# Reads two xyz files that have to at the very least have the same molecular formulae.
# E.g. Structures of product and educt. Based on the pairwise distance matrix the indexing of 
# the two structures is compared to map atoms of e.g. the educt to the corresponding atom in the
# product structure. 
#===============================================================================================

from source.pairwise_distance import pairwise_distance_matrix
from source.xyz_parser import XYZ_reader
import pandas as pd
import numpy as np
import itertools

#===============================================================================================

def atom_mapping_permutation(educt_path,product_path,inverse = True, unit = True, exp =None):
    '''Compute the pairwise distance matrix of a molecule (dataframe object) with x,y,z columns
    param:
        educt_path:     dataframe containing at the least 3 col wit 'x', 'y', 'z' 
        product_path:   dataframe containing at the least 3 col wit 'x', 'y', 'z' 
        inverse:        compute inverse for the pairwise distance matrix (default: False) 
        exp:            compute inverse with exponent for faster decay (default: None)  
        unit:           normalize columns to unit vectors (default: False)
    '''

    # load educt + product and compute the respective pairwise distance matrix
    educt   = pairwise_distance_matrix(XYZ_reader(educt_path)  , inverse = inverse, unit = unit, exp=exp)
    product = pairwise_distance_matrix(XYZ_reader(product_path), inverse = inverse, unit = unit, exp=exp)


    # get all possible permutations
    permutations = list(itertools.permutations(product.index))


    # transform to numpy for easy suffeling
    prod = product.to_numpy()

    # initiate list of scores
    scores = []

    # iterate all permutations
    for p in permutations:

        # compute score between educt and a given index permutation of the product
        # the score is defined as the sum of diagonal elements of the dot product of the two matrices
        score = np.dot(educt, prod[p,:][:,p].transpose()).diagonal().sum()

        # append score to list of scores
        scores.append(score)
 


    # get the set of scores and sort by highest score
    scores = frozenset(scores)
    scores = sorted(scores, reverse=True)

    # iterate over and return the five highest scored permutations
    for position,score in enumerate(scores[:5]):
            print(f"{position +1}: {permutations[scores.index(scores[position])]} (score: {round(score,4)}) ")

def atom_mapping_restricted_permutation(educt_path,product_path,n=2,inverse = True, unit = True, exp =1):
    '''Compute the pairwise distance matrix of a molecule (dataframe object) with x,y,z columns
    param:
        educt_path:     dataframe containing at the least 3 col wit 'x', 'y', 'z' 
        product_path:   dataframe containing at the least 3 col wit 'x', 'y', 'z' 
        n:              number of top picks to consider for each position (default: 2)
        inverse:        compute inverse for the pairwise distance matrix (default: False) 
        exp:            compute inverse with exponent for faster decay (default: None)  
        unit:           normalize columns to unit vectors (default: False)
    '''

    # load educt + product and compute the respective pairwise distance matrix
    product_file = XYZ_reader(product_path)
    educt   = pairwise_distance_matrix(XYZ_reader(educt_path)  , inverse = inverse, unit = unit, exp=exp)
    product = pairwise_distance_matrix(XYZ_reader(product_path), inverse = inverse, unit = unit, exp=exp)

    # compute similarity scores between educt and product
    scores = pd.DataFrame(np.dot(educt, product.transpose()))

    # get the reference which atoms can get which places
    reference = [product_file.loc[product_file['Element'] == element].index.to_list() for element in product_file['Element'].unique()]

    print(n ** product_file.shape[0])
    # initiate options vector
    options = list(range(product_file.shape[0]))
    for i in reference:
        for column in i :
            # fill the options vector at each position i with the nlargest possibilities for that
            options[column] = scores[column].iloc[i].nlargest(n).index.to_list() 


    if len(set(list(itertools.chain.from_iterable(options)))) != product_file.shape[0]:
        raise ValueError('Some atom(s) is missing in the options')
    if n ** product_file.shape[0] > 10**9:
        raise ValueError(f'{n ** product_file.shape[0]} possible permutations are too many!')
  
    # all possible permutations
    permutations = (p for p in product_s(* options) if len(set(p)) == len(p))

    # transform to numpy for easy suffeling
    prod = product.to_numpy()

    # initiate list of scores
    scores = []
    combination = []

    # iterate all permutations
    for p in permutations:
        
        # compute score between educt and a given index permutation of the product
        # the score is defined as the sum of diagonal elements of the dot product of the two matrices
        score = np.dot(educt, prod[p,:][:,p].transpose()).diagonal().sum()

        scores.append(score)
        combination.append(p)
 


    # get the set of scores and sort by highest score
    scores = frozenset(scores)
    scores = sorted(scores, reverse=True)

    # iterate over and return the five highest scored permutations
    for position,score in enumerate(scores[:5]):
            print(f"{position +1}: {combination[scores.index(scores[position])]} (score: {round(score,4)}) ")
