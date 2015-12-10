__author__ = 'Akshay'

import numpy as np
import sklearn.cluster as c
import csv
import time
start_time = time.time()

def importData(filename):
    data = np.genfromtxt(filename, delimiter='\t')
    return data

def importText(filename):
    with open(filename,'r') as f:
        labels = f.read().splitlines()
    labels=[x.strip(' ') for x in labels]
    return labels

def getInd(user_labels,labels):
    ind = []
    for l in user_labels:
        ind.append(labels.index(l))
    return ind


def main():
    labels = importText('data_labels.txt')
    user_labels = importText('user_labels.txt')
    ind = getInd(user_labels,labels)

    test_data = np.genfromtxt('./Data/sampleData.csv')

    test_data = test_data[ind,:]

    kmeans =  c.KMeans(n_clusters=len(ind))


main()