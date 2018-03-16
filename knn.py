#KNN algorithm : supervised classification algorithm

#labelled_data should be a matrix with features x1,...,xn and labels y
#new_example is a vector with features x1,...,xn

#dependencies
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing,cross_validation
import os

df=pd.read_csv('data/data.csv',sep=";")

X=np.array(df.drop(['left'],1))
X=preprocessing.normalize(X)
y=np.array(df['left'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

new_individual=np.array([0.10,0.1,10,150,7,0,0])

#examples
ind1=np.array([3,3])
ind2=np.array([9,0])
labelled_data=np.vstack((ind1,ind2))

new_example=np.array([14,1])

#calculate the distance between 2 individuals
def distance(ind1,ind2):
	distance=np.sum((ind1-ind2)**2)
	return distance

def kdistance(new_example,labelled_data):
	dist=[]
	for i in list(range(0,len(labelled_data))):
		print(i)
		dist.append(distance(new_example,labelled_data[i]))
	return dist

def knn(kdistance_table):
	enum=np.argmin(kdistance_table)
	prediction=X_train[enum]
	print("The new individual is close to ",prediction)

def vote():
	pass

#Calcul of distances
kdistance_table=kdistance(new_individual,X_train)

#Knn
knn(kdistance_table)















