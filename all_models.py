# Logistic regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=sns.load_dataset('titanic')
# print(df.head())
# print(df.shape)
df.drop_duplicates(inplace=True)
df.drop(columns=['deck','adult_male','embark_town','who','class'],axis=1,inplace=True)

df['age'].fillna(df['age'].mean(),inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0],inplace=True)
# print(df.info())

X=df.drop(columns=['survived'],axis=1)
y=df['survived']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
column=['sex','embarked','alive','alone']
Xlabel = X
for i in column:
    Xlabel[i]=le.fit_transform(Xlabel[i])
# print(Xlabel)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scaling_col=['age','fare']
Xlabel[scaling_col]=scalar.fit_transform(Xlabel[scaling_col])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model=LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy= accuracy_score(y_test,y_pred)
# print(accuracy)

confusion=confusion_matrix(y_test,y_pred)
# print(confusion)

report=classification_report(y_test,y_pred)
# print(report)


# KNN
X_train_enc=scalar.fit_transform(X_train)
X_test_enc=scalar.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn_model= KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_enc,y_train)
knn_y_pred= knn_model.predict(X_test_enc)

knn_accuracy= accuracy_score(y_test,knn_y_pred)
# print(knn_accuracy)
knn_confusion=confusion_matrix(y_test,knn_y_pred)
# print(knn_confusion)
knn_report=classification_report(y_test,knn_y_pred)
# print(knn_report)

# Naive bayes
from sklearn.naive_bayes import GaussianNB
model_nb=GaussianNB()
model.fit(X_train,y_train)
y_pred_nb= model.predict(X_test)

nb_accuracy= accuracy_score(y_test,y_pred_nb)
# print(nb_accuracy)
nb_confusion=confusion_matrix(y_test,y_pred_nb)
# print(nb_confusion)
nb_report=classification_report(y_test,y_pred_nb)
# print(nb_report)

# Decision tree
from sklearn.tree import DecisionTreeClassifier

model_dt=DecisionTreeClassifier()
model_dt.fit(X_train_enc,y_train)
y_pred_dt= model_dt.predict(X_test_enc)

dt_accuracy= accuracy_score(y_test,y_pred_dt)
# print(dt_accuracy)
dt_confusion=confusion_matrix(y_test,y_pred_dt)
# print(dt_confusion)
dt_report=classification_report(y_test,y_pred_dt)
# print(dt_report)

# SVM
from sklearn.svm import SVC
model_svm=SVC(kernel='rbf')
model_svm.fit(X_train_enc,y_train)
y_pred_svm= model_svm.predict(X_test_enc)

svm_accuracy= accuracy_score(y_test,y_pred_svm)
# print(svm_accuracy)
svm_confusion=confusion_matrix(y_test,y_pred_svm)
# print(svm_confusion)
svm_report=classification_report(y_test,y_pred_svm)
# print(svm_report)

# K fold cross validation
from sklearn.model_selection import cross_val_score

model_k= cross_val_score(model_dt,X_train_enc,y_train,cv=5,scoring='accuracy')
print(model_k.mean())