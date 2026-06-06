import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr,chi2_contingency

df= pd.read_csv("insurance.csv")

df.drop_duplicates(inplace=True)
# print(df.isnull().sum())

# print(df.dtypes)
df.rename(columns={'sex':'is_male','smoker':'is_smoker'},inplace=True)
# print(df["sex"].value_counts())
df["is_male"]=df["is_male"].map({'male':1,'female':0})
df["is_smoker"]=df["is_smoker"].map({'yes':1,'no':0})

df=pd.get_dummies(df,columns=['region'])
df=df.astype(int)


# print(df.columns)
cols=['age','bmi','children']
scale=StandardScaler()

df[cols]=scale.fit_transform(df[cols])
# print(df.head())    

# pearson correlation calculation
selected_featured=['age', 'is_male', 'bmi', 'children', 'is_smoker',
       'region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest']

correlation= {
    feature :  pearsonr(df[feature], df['charges'])[0]  #pearsonr measure relationship btw two numeric variable, store only correlation value not p_value and it store result in dict
    for feature in selected_featured
}
corr_df= pd.DataFrame(list(correlation.items()), columns=['feature','pearson correlation'])
corr_df= corr_df.sort_values(by='pearson correlation', ascending=False)
print(corr_df)

# chi2
alpha=0.05
cat_feature=['is_male','is_smoker','region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest']

df['charges_bins']= pd.qcut(df['charges'],q=4,labels=False)
chi2_result={}

for col in cat_feature:
    contangency=pd.crosstab(df[col],df['charges_bins'])
    chi2_test,p_value,_,_ = chi2_contingency(contangency)
    decision="reject null(keep)" if p_value < alpha else "accept null(remove)"
    chi2_result[col]={
        "chi2_statistics" : chi2_test,
        "p_value" : p_value,
        "decision" : decision
    }
chi2_df=pd.DataFrame(chi2_result).T
chi2_df=chi2_df.sort_values(by='p_value')
print(chi2_df)
final_df=df[['age','is_male','bmi','children','is_smoker','charges','region_southeast']]
# print(final_df)

from sklearn.model_selection import train_test_split

X=final_df.drop('charges',axis=1)
y=final_df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
r2= r2_score(y_test,y_pred)
# print(r2)

n=X_test.shape[0]
p=X_test.shape[1]

a_r2 = 1-((1-r2)* (n-1)) / (n-p-1)
# print(a_r2)