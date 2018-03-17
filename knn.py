#KNN algorithm : supervised classification algorithm

#labelled_data should be a matrix with features x1,...,xn and labels y
#new_example is a vector with features x1,...,xn

#dependencies
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing,model_selection
from collections import Counter
import scipy
import os

df=pd.read_csv('data/data.csv',sep=";")
print(df)
X=np.array(df.drop(['left'],1))
X_normed=preprocessing.normalize(X)
y=np.array(df['left'])

X_train,X_test,y_train,y_test=model_selection.train_test_split(X_normed,y,test_size=0.2)

new_individual=np.array([0.45,0.47,2,160,3,0,0])

#examples
ind1=np.array([3,3])
ind2=np.array([9,0])
labelled_data=np.vstack((ind1,ind2))

new_example=np.array([14,1])

#calculate the distance between 2 individuals
def distance(ind1,ind2):
	distance=np.sum((ind1-ind2)**2)
	return distance

#create the table if distances between the new example and every other labelled data
def kdistance(new_example,labelled_data):
	dist=[]
	for i in list(range(0,len(labelled_data))):
		dist.append(distance(new_example,labelled_data[i]))
	return dist

def knn(k,kdistance_table):
	enum=np.argsort(kdistance_table)[:k]
	predictions=y_train[enum]
	c=Counter(predictions)
	key=[element for element, count in c.most_common(1)]
	return key[0]

def accuracy(X_test,y_test,X_train):
	total=len(y_test)
	liste=[]
	for example in range(len(X_test)):
		prediction=knn(5,kdistance(X_test[example],X_train))
		liste.append(prediction)
	good_predictions=np.sum(liste==y_test)
	accuracy=good_predictions/total
	return accuracy


# #Calcul of distances
# kdistance_table=kdistance(new_individual,X_train)

# #Knn
# knn(5,kdistance_table)

#Confusion matrix
# print("accuracy of algorithm is",accuracy(X_test,y_test,X_train))


#data analysis
analysis=True
if analysis==True:
	stats=scipy.stats.describe(X)
	print(stats.nobs)
	print(stats.minmax)
	print(stats.mean)
	print(stats.variance)
	print(stats.skewness)
	time_spend_hist=plt.hist(df['time_spend_company'])
	plt.title('time_spend_company')
	plt.show()
	montly_hours_hist=plt.hist(df['average_montly_hours'])
	plt.title('average_montly_hours')
	plt.show()
	number_project_hist=plt.hist(df['number_project'])
	plt.title('number_project')
	plt.show()
	number_project_hist=plt.hist(df['promotion_last_5years'])
	plt.title('promotion_last_5years')
	plt.show()
	number_project_hist=plt.hist(df['left'])
	plt.title('skewness of the data')
	plt.show()















