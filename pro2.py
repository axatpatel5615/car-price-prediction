import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,chi2_contingency

df= pd.read_csv(r"C:\Users\Patel Axat\Desktop\python\ML\car_price_prediction.csv")

real_col=['ID', 'Price', 'Levy', 'Manufacturer', 'Model', 'Prod. year',
       'Category', 'Leather interior', 'Fuel type', 'Engine volume', 'Mileage',
       'Cylinders', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color',
       'Airbags']

pd.set_option('display.max_columns', None)
df.drop_duplicates(inplace=True)
# print(df.head())
# print(df.shape)
# df.info()
# print(df.describe())
# print(df.isnull().sum())
# print(df.shape)

df['Levy']=pd.to_numeric(df['Levy'], errors='coerce')
df['Levy'].fillna(df['Levy'].mean(),inplace=True)
# print(df['Levy'].isnull().sum())

df['Engine volume']=df['Engine volume'].str.replace(r'\D','',regex=True)
df['Engine volume']=pd.to_numeric(df['Engine volume'])
# print(df['Engine volume'].dtype)

df['Mileage']=df['Mileage'].str.replace(r'\D','',regex=True)
df['Mileage']=pd.to_numeric(df['Mileage'])
# print(df['Mileage'].dtype)

# print(df['Doors'].value_counts())
df['Doors']=df['Doors'].replace({
    '04-May' : 5,
    '02-Mar' : 3,
    '>5' : 6
})
# print(df['Doors'].dtype)

# print(df['Drive wheels'].value_counts())
df['Drive wheels']=df['Drive wheels'].replace({
    'Front' : 1,
    'Rear' : 2,
    '4x4' : 4
})
# print(df['Doors'].dtype)


df = df[df['Price'] < df['Price'].quantile(0.99)]
# sns.histplot(df['Price'],kde=True,bins=10)

# sns.boxplot(data=df,x='Cylinders',y='Price')
# plt.xticks(rotation=90)

# sns.countplot(x='Cylinders',data=df)

# sns.scatterplot(data=df,x='Cylinders', y='Price')

# sns.barplot(data=df, x='Model', y='Price')
# plt.xticks(rotation=90)

# sns.heatmap(df.corr(),annot=True)
# plt.tight_layout()
# plt.show()

# print(df['Drive wheels'].value_counts())
X=df.drop(columns=['Price'],axis=1)
y=df['Price']

col=['Manufacturer','Model','Category','Leather interior','Fuel type','Gear box type','Drive wheels','Wheel','Color']
enc_col=pd.get_dummies(X,columns=col)
# print(enc_col.head(3))

corr_series = enc_col.corrwith(df['Price'])
# Convert to DataFrame and sort
corr_df = corr_series.to_frame(name='pearson correlation').reset_index()
corr_df.columns = ['feature', 'pearson correlation']
corr_df = corr_df.sort_values(by='pearson correlation', ascending=False)
# print(corr_df.shape)

alpha=0.05
encoded_column=[col for col in enc_col if col not in real_col ]
# print(encoded_column)
df['Price_bins']=pd.qcut(df['Price'],q=4,labels=False)
chi2_result={}

for col in encoded_column:
    contangency=pd.crosstab(enc_col[col],df['Price_bins'])
    chi2_test,p_value,_,_=chi2_contingency(contangency)
    decision= "reject null(keep)" if p_value < alpha else "accept null(remove)"
    chi2_result[col]={
        'chi2_statistics' : chi2_test,
        'p_value' : p_value,
        'decision' : decision
    }

chi2_df=pd.DataFrame(chi2_result).T
chi2_df = chi2_df.sort_values(by='p_value')
# print(chi2_df)

col_remove=chi2_df[chi2_df['decision']=='accept null(remove)'].index.tolist()
col_keep=chi2_df[chi2_df['decision']=='reject null(keep)'].index.tolist()
# print(len(col_keep))
enc_colu=enc_col.drop(columns=col_remove)
# print(enc_colu.columns)

from sklearn.preprocessing import LabelEncoder

lab_col=['Manufacturer','Model','Category','Leather interior','Fuel type','Gear box type','Drive wheels','Wheel','Color']
enc= LabelEncoder()

Xlabel = X
for i in lab_col:
    Xlabel[i]=enc.fit_transform(Xlabel[i])
# print(Xlabel)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

non_bin_enc=[non_bin_col for non_bin_col in enc_colu
             if not set(enc_colu[non_bin_col].unique())<={0,1,2,3}]
print(non_bin_enc)
colu=['ID', 'Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Airbags']

enc_colu[colu] = scaler.fit_transform(enc_colu[colu])

nb_enc=[nb_col for nb_col in Xlabel
             if not set(Xlabel[nb_col].unique())<={0,1,2,3}]

sca_col=['ID', 'Levy', 'Manufacturer', 'Model', 'Prod. year', 'Category', 'Fuel type', 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Color', 'Airbags']
Xlabel[sca_col] = scaler.fit_transform(Xlabel[sca_col])
# print(Xlabel)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(enc_colu, y, test_size=0.20, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
r2= r2_score(y_test,y_pred)
print(r2)

n=enc_colu.shape[0]
p=enc_colu.shape[1]

a_r2= 1-((1-r2)*(n-1)) / (n-p-1)
print(a_r2)


X_train, X_test, y_train, y_test = train_test_split(Xlabel, y, test_size=0.20, random_state=42)
model2=LinearRegression()
model2.fit(X_train,y_train)

y_pred=model2.predict(X_test)
r22= r2_score(y_test,y_pred)
# print(r22)
n=Xlabel.shape[0]
p=Xlabel.shape[1]

a_r2= 1-((1-r2)*(n-1)) / (n-p-1)
# print(a_r2)

