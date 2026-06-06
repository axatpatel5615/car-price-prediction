import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

df=pd.read_csv('heart.csv')
df.drop_duplicates(inplace=True)

X=df.drop(columns=['target'],axis=1)
y=df['target']

models={
    'Logistic regression' : LogisticRegression(),
    'KNN' : KNeighborsClassifier(),
    'Naive bayes' : GaussianNB(),
    'Decision Tree' : DecisionTreeClassifier(),
    'SVM' : SVC()
}

scaler=StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train_scal=scaler.fit_transform(X_train)
X_test_scal=scaler.fit_transform(X_test)

result=[]
for name, model in models.items():
    model.fit(X_train_scal,y_train)
    y_pred=model.predict(X_test_scal)
    acc=accuracy_score(y_test,y_pred)
    F1=f1_score(y_test,y_pred)
    result.append({
        'model' : name,
        'accuracy' : acc,
        'f1_score' : F1
    })

df1=pd.DataFrame(result)
# print(df1)

import joblib
joblib.dump(models['Naive bayes'],'Naive_bayes.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(X.columns.tolist(),'columns.pkl')