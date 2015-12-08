nb__author__ = 'Akshay Ajbani, Emily Misnick, Lucas Neves'


import os
import numpy as np
import sys
import math
import csv
from sklearn.svm import SVC
import time
start_time = time.time()

num_rows = 38294
test_size = 12	#Change this to change how big the test sample size is
NEWDATA=True # if making tweaks that dont alter file I/o, can flag this false to speed up runtime 
def importData(filename):
	with open(filename) as f:
		ncols = len(f.readline().split('\t')) #gets number of columns
	data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(4,ncols))
	ncols = np.size(data,1)

	# train_data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(4,ncols-test_size)) #skips text containing columns and rows to load only expression data
	# test_data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(ncols-test_size,ncols)) #skips text containing columns and rows to load only expression data
	#train_data = train_data[:num_rows]
	#test_data = test_data[:num_rows]
	return data, ncols

def splitData(data,ncols):

	test_cols = np.random.choice(ncols, test_size, replace=False)
	test_data = data[:,test_cols]
	train_data = np.delete(data,test_cols,1)

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
	clf = SVC(probability= True,decision_function_shape='ovr',random_state=np.random.randint(10000),kernel="linear")
	clf.fit(data,labels)
	return clf

def loadTestData(filename):
	with open(filename,'r') as f:
		data=np.loadtxt(filename,delimiter="\t",unpack=False)
	return data
def loadTestLabels(filename):
	with open(filename,'r') as f:
		labels = f.read().splitlines()
	labels=[x.strip(' ') for x in labels]
	return labels


def sortData(data1,data2,data3,protein_labels,test_data_Labels,test_data): #extremely time consuming

	data_out1=[]
	data_out1=np.array(data_out1)
	data_out2=[]
	data_out2=np.array(data_out2)
	data_out3=[]
	data_out3=np.array(data_out3)
	data_out_labels=[]

	todel=[]
	init=True
	for i in range(len(test_data_Labels)): 
		isfound=True
		try:
			matchlist = protein_labels.index(test_data_Labels[i])
		except ValueError:
			isfound=False

		if isfound==True:
			data_out_labels.append(protein_labels[matchlist])

			if init==True:
				data_out1=data1[matchlist]
				data_out2=data2[matchlist]
				data_out3=data3[matchlist]
				out_data=test_data[i]
				init=False
			else:
				data_out1=np.vstack((data_out1,data1[matchlist]))
				data_out2=np.vstack((data_out2,data2[matchlist]))
				data_out3=np.vstack((data_out3,data3[matchlist]))
				out_data=np.vstack((out_data,test_data[i]))				

		if isfound==False:
			isfound=True

	print(len(out_data))
	print(len(data_out1))
	return [data_out1,data_out2,data_out3,data_out_labels,out_data]

def saveData(data,test):
	class1=[]
	class1=np.array(class1)
	class2=[]
	class2=np.array(class2)
	class3=[]
	class3=np.array(class3)

	for i in range(0,len(data)):
		if data[i]==1:
			class1=np.append(class1,test[i,:])
		elif data[i]==2:
			class2=np.append(class2,test[i,:])
		elif data[i]==3:
			class3=np.append(class3,test[i,:])
	class1=np.transpose(class1)
	class2=np.transpose(class2)
	class3=np.transpose(class3)

	np.savetxt('test_results_trial_class1.csv', class1, delimiter="\t")
	np.savetxt('test_results_trial_class2.csv', class2, delimiter="\t")
	np.savetxt('test_results_trial_class3.csv', class3, delimiter="\t")


def main():
	num_trials = 1
	runDataTest = True
	filename1 = "./Data/G1_singlecells_counts.txt"
	filename2 = "./Data/G2M_singlecells_counts.txt"
	filename3 = "./Data/S_singlecells_counts.txt"
	testname = "./Data/sampleData.csv"
	testlabelname = "./Data/sampleDataLabels.txt"
	train_labels = []
	test_labels = []
	errors = 0

	protein_labels = importProteinLabels(filename1)
	data1, ncols1 = importData(filename1)
	data2, ncols2 = importData(filename2)
	data3, ncols3 = importData(filename3)


	if(runDataTest ==True):
		test_data=loadTestData(testname)
		test_data_labels=loadTestLabels(testlabelname)

		[trimmed_data1,trimmed_data2,trimmed_data3,trimmed_labels,test_data] =sortData(data1,data2,data3,protein_labels,test_data_labels,test_data)

		print(str(len(trimmed_data1)))

		data1=trimmed_data1
		data2=trimmed_data2
		data3=trimmed_data3
		protein_labels=trimmed_labels


	for trial in range(0,num_trials):
		# Data 1
		train_data1, test_data1 = splitData(data1,ncols1)
		for i in range(0,len(train_data1[1])):
			train_labels.append(1)
		for i in range(0,len(test_data1[1])):
			test_labels.append(1)
		train_data1 = train_data1.transpose()
		test_data1 = test_data1.transpose()


		# Data 2
		train_data2, test_data2 = splitData(data2,ncols2)
		for i in range(0,len(train_data2[1])):
			train_labels.append(2)
		for i in range(0,len(train_data2[1])):
			test_labels.append(2)
		train_data2 = train_data2.transpose()
		test_data2 = test_data2.transpose()

		train_data= np.vstack((train_data1,train_data2))

		# Data 3
		train_data3, test_data3 = splitData(data3,ncols3)
		for i in range(0,len(train_data3[1])):
			train_labels.append(3)
		for i in range(0,len(train_data3[1])):
			test_labels.append(3)
		train_data3 = train_data3.transpose()
		test_data3 = test_data3.transpose()

		train_data= np.vstack((train_data,train_data3))

		clf=TrainSVM(train_data,train_labels)
	#	print(train_labels)

		predict1 = clf.predict(test_data1)
		predict2 = clf.predict(test_data2)
		predict3 = clf.predict(test_data3)
		if runDataTest==True:
			print(str(len(test_data)))
			test_data=test_data.transpose()
			predict_data=clf.predict(test_data)
			saveData(predict_data,test_data)

		errors += (np.count_nonzero(predict1-1)+np.count_nonzero(predict2-2)+np.count_nonzero(predict3-3))

	if runDataTest == True:
		with open('test statistics.txt', 'w') as st:
			st.write('Average Errors after'+ str(num_trials) + ' trials: \n')
			st.write(errors/num_trials + '\n')
			st.write('% Errors: '+(errors/(num_trials*test_size))+'\n')
			st.close()

	print('Average Errors after ' + str(num_trials) + ' trials:')
	print(errors/num_trials)
	print('% Errors: ')
	print((errors/(num_trials*test_size))*100)


main()
print("--- %s minutes ---" % ((time.time() - start_time)/60))