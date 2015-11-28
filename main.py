__author__ = 'Akshay Ajbani, Emily Misnick, Lucas Neves'


import os
import numpy as np
import sys
import random
import math
import csv

def importData(filename):
    with open(filename) as f:
        ncols = len(f.readline().split('\t')) #gets number of columnts
    data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(4,ncols)) #skips text containing columns and rows to load only expression data
    return data
def importProteinLabels(filename):
	labels=[]
	with open(filename,'r') as f:
		next(f)
		reader=csv.reader(f,delimiter='\t')
		for rows in reader:
			labels.append(rows[2])
	return labels
def TrainSVM(data,protein_labels,label):


def main():
	filename = "./Data/G1_singlecells_counts.txt"
    data = importData(filename)
    protein_labels = importProteinLabels(filename)

main()