import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('minute_weather.csv')
df.dropna()
# print(df)
df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
df.head()
df.shape


# df = df[df['EPS'].notna()]

sample_df = df[(df['rowID']%10) == 0]
sample_df.head()

sample_df1 = sample_df.drop(columns =['rain_accumulation','rain_duration'])
print(sample_df1.columns)

cols_of_interest = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed','max_wind_direction','max_wind_speed', 'relative_humidity']

data = sample_df1[cols_of_interest]
# print(np.where(np.isnan(data)))

data.head()


X = StandardScaler().fit_transform(data)
# X
# remove all the nan values from 2D array
X = X[~np.isnan(X).any(axis=1)]

print(np.all(np.isfinite(X)))
# np.nan_to_num(X)
# print(np.all(np.isfinite(X)))
# print(np.where(np.isnan(X)))

#Set number of clusters at initialisation time
k_means = KMeans(n_clusters = 12)
#Run the clustering algorithm
model = k_means.fit(X)
# model
#Generate cluster predictions and store in y_hat
y_hat = k_means.predict(X)

print(y_hat)
print(len(y_hat))

labels = k_means.labels_

# scores for measuring accuracy of k means clustering
print(metrics.silhouette_score(X, labels, metric = 'euclidean'))
print(metrics.calinski_harabasz_score(X, labels))
