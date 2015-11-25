import os
import numpy as np
import sys
import random
import math


def main():
	with open("G1_singlecells_counts.txt") as f:
		ncols = len(f.readline().split('\t')) #gets number of columnts
	data = np.loadtxt('G1_singlecells_counts.txt', delimiter="\t",unpack=False,skiprows=1,usecols=range(4,ncols)) #skips text containing columns and rows to load only expression data


main()