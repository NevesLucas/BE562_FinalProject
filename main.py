__author__ = 'Akshay Ajbani, Emily Misnick, Lucas Neves'


import os
import numpy as np
import sys
import math
import csv
from sklearn.svm import SVC

num_rows = 38294

def importData(filename):
	with open(filename) as f:
		ncols = len(f.readline().split('\t')) #gets number of columnts
	train_data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(4,ncols-16)) #skips text containing columns and rows to load only expression data
	test_data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(80,ncols)) #skips text containing columns and rows to load only expression data
	train_data = train_data[:num_rows]
	test_data = test_data[:num_rows]
	return train_data, test_data


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
	SVC(probability= True,random_state=22003)
	clf.fit(data,labels)
	return clf


def main():
	filename1 = "./Data/G1_singlecells_counts.txt"
	filename2 = "./Data/G2M_singlecells_counts.txt"
	filename3 = "./Data/S_singlecells_counts.txt"
	train_labels = []
	test_labels = []

	protein_labels = importProteinLabels(filename1)

	# Data 1
	train_data1, test_data1 = importData(filename1)
	for i in range(0,train_data1[1]):
		train_labels.append(1)
	for i in range(0,test_data1[1]):
		test_labels.append(1)
	train_data1 = train_data1.transpose()
	test_data1 = test_data1.transpose()

	# Data 2
	train_data2, test_data2 = importData(filename2)
	for i in range(0,train_data2[1]):
		train_labels.append(1)
	for i in range(0,test_data2[1]):
		test_labels.append(1)
	train_data2 = train_data2.transpose()
	test_data2 = test_data2.transpose()

	# Data 3
	train_data3, test_data3 = importData(filename3)
	for i in range(0,train_data3[1]):
		train_labels.append(1)
	for i in range(0,test_data3[1]):
		test_labels.append(1)
	train_data3 = train_data3.transpose()
	test_data3 = test_data3.transpose()








main()