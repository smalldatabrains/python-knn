import numpy as np
from sklearn import preprocessing,model_selection,neighbors
from sklearn.manifold import TSNE
import pandas as pd
from ggplot import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df=pd.read_csv('data/data.csv',sep=";")

X=np.array(df.drop(['left'],1))
X=preprocessing.normalize(X)
y=np.array(df['left'])

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier() #k default value is 5
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)




new_example=np.array([0.10,0.1,10,150,7,0,0])
new_example=new_example.reshape(1,-1)
prediction=clf.predict(new_example)


if (prediction==1):
	p="leave the company"
else:
	p="stay in the company"
print("For individual with characteristics",new_example,"our prediction is that he will", p)

# #tsne with sklearn
# X=pd.DataFrame(X)
# tsne=TSNE(n_components=3,learning_rate=10)
# tsne_results=tsne.fit_transform(X)

# df_tsne=X.copy()
# df_tsne['x-tsne']=tsne_results[:,0]
# df_tsne['y-tsne']=tsne_results[:,1]
# df_tsne['z-tsne']=tsne_results[:,2]
# df_tsne['label']=y

# # chart=ggplot(df_tsne,aes(x='x-tsne',y='y-tsne',color='label'))+geom_point(size=70,alpha=0.1)+ggtitle("TSNE dimensions colored by group")
# # print(chart)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df_tsne['x-tsne'],df_tsne['y-tsne'],df_tsne['z-tsne'],c=df_tsne['label'])
# plt.show()