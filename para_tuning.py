# Grid search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=sns.load_dataset('iris')
# print(df.head())

# print(df['species'].unique())

X=df.drop(columns=['species'],axis=1)
y=df['species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier()
model_knn.fit(X_train,y_train)
# print(model_knn.score(X_test,y_test))

from sklearn.model_selection import GridSearchCV
modelcv=GridSearchCV((model_knn),{
    'n_neighbors' : [5,15,25],
    'weights' : ['uniform', 'distance'],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
},cv=5,return_train_score=False)
modelcv.fit(X,y)

# print(modelcv.cv_results_)
result=pd.DataFrame(modelcv.cv_results_)
# print(result[['param_n_neighbors','param_algorithm', 'param_weights','mean_test_score']]) 

# Random search 
from sklearn.model_selection import RandomizedSearchCV

model_rs= RandomizedSearchCV((model_knn),{
    'n_neighbors' : [5,15,25],
    'weights' : ['uniform', 'distance'],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
},n_iter=8,cv=5,return_train_score=False)
model_rs.fit(X,y)
result_rs=pd.DataFrame(model_rs.cv_results_)
# print(result_rs[['param_n_neighbors','param_algorithm', 'param_weights','mean_test_score']])
