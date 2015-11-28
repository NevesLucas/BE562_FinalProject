__author__ = 'Akshay Ajbani, Emily Misnick, Lucas Neves'


import os
import numpy as np
import sys
import math
import csv
from sklearn.svm import SVC

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

def TrainSVM(data,labels):
	clf = SVC()
	SVC(probability=true,random_state=22003)
	clf.fit(data,Labels)
	return clf


def main():
	filename1 = "./Data/G1_singlecells_counts.txt"
	filename2 = "./Data/G2M_singlecells_counts.txt"
	filename3 = "./Data/S_singlecells_counts.txt"
	labels[]

	protein_labels = importProteinLabels(filename1)
	
	data1 = importData(filename1)
	for i in range(0,data1[1]):
		labels.append(1)
	data1=data1.transpose()

	data2 = importData(filename2)
	for i in range(0,data2[1]):
		labels.append(2)
	data2=data2.transpose()

	data3 = importData(filename3)
	for i in range(0,data3[1]):
		labels.append(3)
	data3=data3.transpose()




main()