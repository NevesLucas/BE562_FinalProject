import numpy as np
import csv

def importData(filename):
	with open(filename) as f:
		ncols = len(f.readline().split('\t')) #gets number of columnts
	data = np.loadtxt(filename, delimiter="\t",unpack=False,usecols=range(1,ncols))
	#ncols = np.size(data,1)

	# train_data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(4,ncols-test_size)) #skips text containing columns and rows to load only expression data
	# test_data = np.loadtxt(filename, delimiter="\t",unpack=False,skiprows=1,usecols=range(ncols-test_size,ncols)) #skips text containing columns and rows to load only expression data
	#train_data = train_data[:num_rows]
	#test_data = test_data[:num_rows]
	return data

def importProteinLabels(filename):
	labels=[]
	with open(filename,'r') as f:
		next(f)
		reader=csv.reader(f,delimiter='\t')
		for rows in reader:
			labels.append(rows[0])
	return labels

def main():
	stacked_data=[]
	labels=importProteinLabels("./Data/GSM1377612_RPKM_mESC1.txt")
	for i in range(1,63):
		filename = "./Data/GSM13776"+str(11+i)+"_RPKM_mESC"+str(i)+".txt"
		data = importData(filename)
		data.shape
		if i <2:
			stacked_data=data
			
		else:
			stacked_data=np.vstack((stacked_data,data))
	stacked_data=np.transpose(stacked_data)
	np.savetxt("sampleData.csv", stacked_data, delimiter="\t")
	data = open("sampleDataLabels.txt", "w")
	for c in labels:
		data.write("%s\n" % c)
	data.close()
main()
