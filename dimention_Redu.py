import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# PCA
scalar= StandardScaler()
X, y = make_blobs(n_samples=250, centers=3, n_features=4,random_state=0)
X = scalar.fit_transform(X)

pca = PCA(n_components=2)
pca_model = pca.fit_transform(X)
df = pd.DataFrame(pca_model, columns=['PC1', 'PC2'])
df['target'] = y
sns.scatterplot(x=df['PC1'], y=df['PC2'],hue=df['target'], palette='deep')
plt.show()