#KNN algorithm : supervised classification algorithm

#labelled_data should be a matrix with features x1,...,xn and labels y
#new_example is a vector with features x1,...,xn

#dependencies
import numpy as np
import pandas as pd
import math
import os

#data
data=pd.read_csv("data/data.csv",sep=";")

n_train=round(len(data)*0.70)

train=data[0:n_train]
test=data[n_train+1:len(data)]

print(train.shape,test.shape)

#normalize features

train.iloc[:,0:7]=(train.iloc[:,0:7]-train.iloc[:,0:7].mean())/(train.iloc[:,0:7].max()-train.iloc[:,0:7].min())
test.iloc[:,0:7]=(test.iloc[:,0:7]-test.iloc[:,0:7].mean())/(test.iloc[:,0:7].max()-test.iloc[:,0:7].min())
print(train)
#2 groups of employee, those who left the company and those who stayed

print(train.shape,test.shape)
print(list(train))

group1=train[train.left==1]
group2=train[train.left==0]

print("group1 dimensions:",group1.shape,"group2 dimensions:",group2.shape)

# group1=train.loc[]
# group2=train

X_train=train.iloc[:,0:7]
Y_train=train.iloc[:,-1]
X_test=train.iloc[:,0:7]
Y_test=train.iloc[:,-1]

#examples
ind1=np.array([3,3])
ind2=np.array([9,0])
labelled_data=np.vstack((ind1,ind2))

new_example=np.array([14,1])

#calculate the distance between 2 individuals
def distance(ind1,ind2):
	distance=np.sum((ind1-ind2)**2)
	return distance

def kdistance(k,new_example,labelled_data):
	dist=[]
	for i in list(range(0,k+1)):
		print(i)
		dist.append(distance(new_example,labelled_data[i]))
		print(dist)
	return dist


def knn(kdistance_table):
	enum=np.argmin(kdistance_table)
	prediction=labelled_data[enum]
	print("The new individual is close to ",prediction)

#Calcul of distances
kdistance_table=kdistance(1,new_example,labelled_data)

#Knn
knn(kdistance_table)















