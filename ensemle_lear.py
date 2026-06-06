import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# staking
df=sns.load_dataset('iris')

X=df.drop(columns=['species'],axis=1)
y=df['species']

le=LabelEncoder()
# model_sc=StackingClassifier()
y_enc= le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.20, random_state=42)

base_models = [
    ('DT', DecisionTreeClassifier()),
    ('LR', LogisticRegression()),
    ('SVM', SVC())
]

meta_model = LogisticRegression()

staking= StackingClassifier(
    estimators=base_models,
    final_estimator= meta_model,
    cv=5
)

staking.fit(X_train,y_train)
y_pred = staking.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
# print(accuracy)

# bagging
from sklearn.ensemble import BaggingClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.20, random_state=42)

model_bag= BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100
)

model_bag.fit(X_train,y_train)
y_pred_bag = model_bag.predict(X_test)

accuracy_bag= accuracy_score(y_test,y_pred_bag)
# print(accuracy_bag)

# Boosting
# 1. adaboost
from sklearn.ensemble import AdaBoostClassifier
model_ada= AdaBoostClassifier(estimator=DecisionTreeClassifier(),n_estimators=100)

model_ada.fit(X_train,y_train)
y_pred=model_ada.predict(X_test)

accuracy_ada= accuracy_score(y_test,y_pred)
# print(accuracy_ada)

# 2.Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
model_gb= GradientBoostingClassifier(n_estimators=100,max_depth=3)

model_gb.fit(X_train,y_train)
y_pred_gb= model_gb.predict(X_test)

accuracy_gb= accuracy_score(y_test,y_pred)
# print(accuracy_gb)

# XG boost
from xgboost import XGBClassifier
model_xg=XGBClassifier(n_estimators=100,max_depth=3)
model_xg.fit(X_train,y_train)

y_pred_xg= model_xg.predict(X_test)
accuracy_xg= accuracy_score(y_test,y_pred)
print(accuracy_xg)
