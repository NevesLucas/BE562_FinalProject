__author__ = 'Akshay Ajbani, Emily Misnick, Lucas Neves'


import os
import numpy as np
import sys
import random
import math
def importData(filename):
    filename = "./Data/" + filename
    with open(filename) as f:
        ncols = len(f.readline().split('\t')) #gets number of columnts
    data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(4,ncols)) #skips text containing columns and rows to load only expression data
    return data

def main():
    data = importData("G1_singlecells_counts.txt")

main()