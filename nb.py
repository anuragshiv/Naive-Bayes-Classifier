import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as mpl
from numpy import genfromtxt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import random

def training(df,classes):

	features=df.shape[1]-1
	size=[]
	meanvec=[]
	for i in [classes[0],classes[1]]:
		# df1=df.iloc[hashed[i],:]
		# print df1
		df1=df[df.label==i]
		size.append(df1.shape[0])
		meanvec1=df1.mean(axis=0)	
		meanvec1=meanvec1[:features]
		meanvec.append(np.asarray(meanvec1))

	stdvec=np.asarray(df.std(axis=0))
	stdvec=stdvec[:features]
	ws=list()
	wi=list()
	#w0
	for i in range(len(meanvec[0])):
		ws.append((np.square(meanvec[1][i])-np.square(meanvec[0][i]))/np.square(stdvec[i]))
	w0=np.log(1) + (np.sum(ws)/2)

	for i in range(len(meanvec1)):
		wi.append((meanvec[0][i]-meanvec[1][i])/np.square(stdvec[i]))

	return wi,w0

def testing(test,w,w0):
	
	cumsum=0

	for i in range(len(matrix[0])-1):
		cumsum+=w[i]*test[i]
	p=(np.exp(w0 + cumsum)/(1+np.exp(w0+cumsum)))
	return p,1-p


def get_data():
	df=pd.read_csv("input filepath")
	test=pd.read_csv("output filepath")

	return df,test

if __name__=="__main__":
	classes=[3,5]
	train,test=get_data()
	df9=train.drop('label',1)
	matr=np.asarray(df9)
	matr=StandardScaler().fit_transform(matr)
	for i in range(df9.shape[1]):
		train.ix[:,i]= matr[:,i]
	y=list(test.label)
	w,w0=training(train,classes)
	df1=test.drop('label',1)
	matrix=np.asarray(df1)
	matrix=StandardScaler().fit_transform(matrix)
	ypred=[]
	hashed={'T':0,'F':0}
	for i in range(len(test)):
		p1,p2=testing(matrix[i],w,w0)
		if((p1>p2 and y[i]==classes[0]) or (p1<p2 and y[i]==classes[1])):
			hashed['T']+=1
			ypred.append(y[i])
		else:
			hashed['F']+=1
			if(p1>p2):
				ypred.append(classes[0])
			elif(p1<p2):
				ypred.append(classes[1])
	ypred=np.asarray(ypred)
	for i in range(len(y)):
		print str(y[i]) + "," + str(ypred[i])
	print float(hashed['T'])/(hashed['T']+hashed['F'])*100.0
	print confusion_matrix(y, ypred)


	# test(df,rows)


