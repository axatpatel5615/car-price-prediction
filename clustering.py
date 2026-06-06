import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

# K Means cluster
scalar= StandardScaler()
X, y = make_blobs(n_samples=250, centers=3, n_features=2,random_state=42)
df=pd.DataFrame(X,columns=['feature1','feature2'])

X_scaled= scalar.fit_transform(df)
inertia=[]
K_range= range(1,11)  #check cluster from 1 to 11

for k in K_range:
    model= KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

# plt.plot(K_range, inertia,marker='o') 
# plt.show()  

k_model= KMeans(n_clusters=3,random_state=42)
labels= k_model.fit_predict(X_scaled)
df['cluster'] = labels
# sns.scatterplot(x=df['feature1'],y=df['feature2'],hue=df['cluster'])
# plt.show()  


# DBscan cluster
X1, y = make_moons(n_samples=250, noise=0.05, random_state=42)
from sklearn.cluster import DBSCAN

df= pd.DataFrame(X1,columns=['feature11','feature22'])
df_scaled = scalar.fit_transform(X1)

df_model= DBSCAN(eps=0.25,min_samples=5)
db_model=df_model.fit_predict(df_scaled)
# df['db_cluster']= db_model

sns.scatterplot(
    x=df['feature11'],
    y=df['feature22'],
    hue=db_model,
    palette='deep',
)

plt.tight_layout()
plt.show()
