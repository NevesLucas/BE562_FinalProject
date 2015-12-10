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
    norun=False
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
def importData(filename):
    with open(filename) as f:
        ncols = len(f.readline().split('\t')) #gets number of columns
    data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(4,ncols))
    return data


def main():
    labels = importText('./Data/proteinlabels.txt')

    for i in range(len(labels)):
        labels[i]=labels[i].lower()
    
    user_labels = importText('user_labels.txt')
    for i in range(len(user_labels)):
        user_labels[i]=user_labels[i].lower()

    ind = getInd(user_labels,labels)
    
    test_data = importData('./Data/G2M_singlecells_counts.txt')
    
    test_data = test_data[ind,:]
    
    print(ind)

    test_data=test_data.transpose()
    print(test_data[1])
    
    kmeans =  c.KMeans(n_clusters=5,random_state=180,n_jobs=1,init='k-means++').fit_predict(test_data)
    np.savetxt('G2Mcluster_out.txt',kmeans,delimiter="\t")
    print(kmeans)



main()