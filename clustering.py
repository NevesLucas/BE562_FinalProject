__author__ = 'Akshay'

import numpy as np
import sklearn.cluster as c
import csv
import time
import matplotlib.pyplot as plt
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
        try:
            val=labels.index(l)
        except:
            norun=True
        if norun==True:
            norun=False
        else:
            ind.append(val)
    return ind


def main():
    labels = importText('Data/sampleDataLabels.txt')
    user_labels = importText('user_labels.txt')
    ind = getInd(user_labels,labels)
    test_data = np.genfromtxt('./Data/sampleData.csv')
    test_data = test_data[ind,:]

    kmeans =  c.KMeans(n_clusters=5,random_state=170,n_jobs=1,init='k-means++').fit_predict(test_data)
    np.savetxt('cluster_out.txt',kmeans,delimiter="\t")



main()