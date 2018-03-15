import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd

df=pd.read_csv('data/data.csv',sep=";")

X=np.array(df.drop(['left'],1))
X=preprocessing.normalize(X)
y=np.array(df['left'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)


new_example=np.array([0.10,0.1,10,150,7,0,0])
new_example=new_example.reshape(1,-1)
prediction=clf.predict(new_example)
print(prediction)