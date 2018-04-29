import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import pandas as pd

style.use('ggplot')

df = pd.read_csv('KNN_DATA.csv')
X = df.iloc[:,1:].as_matrix()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

z = np.array([[7.092207670211791992e-01,4.852280020713806152e-01],
              [8.701885938644409180e-01,8.108747005462646484e-01],
              [6.476376652717590332e-01,5.138314366340637207e-01],
              [5.683318376541137695e-01,7.279945611953735352e-01],
              [4.801380634307861328e-01,5.642787218093872070e-01],
              [7.355208396911621094e-01,5.564780235290527344e-01]
             ])

plt.scatter(X[:,0],X[:,1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5);
plt.scatter(z[:,0],z[:,1], c = 'red', s=100)
plt.show()
# cmap = plt.get_cmap('jet')
# colors = cmap(np.linspace(0, 1.0, len(labels)))
# #colors = ["g.","r.","c.","y."]
# for i in range(len(X)):
#     #print("coordinate:",X[i], "label:", labels[i])
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


# plt.scatter(centroids[:, 0],centroids[:, 1], marker = "*", c='#050505')

# plt.show()

