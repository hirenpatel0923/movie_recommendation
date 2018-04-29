import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv('test_user_reviews.csv', encoding='ISO-8859-1')
target = df.iloc[:,1:2]
movies = target.groupby(['movie_id'])
df = df.drop(labels=['movie_id','review_id','review','ratings'], axis=1)
df.insert(0,column='dummy',value=0)
df = df.fillna(0)
train_x = df.as_matrix()
np.savetxt("user_numpy_opt.csv", train_x, delimiter=",")


inputs = tf.placeholder('float', shape=(None,48), name='attributes')

reshape = tf.reshape(inputs, shape=[tf.shape(inputs)[0],8,6])
pool1 = tf.layers.max_pooling1d(inputs=reshape, pool_size=2, strides=1)
pool2 = tf.layers.average_pooling1d(inputs=pool1, pool_size=2, strides=2)
pool3 = tf.layers.average_pooling1d(inputs=pool2, pool_size=2, strides=2)
reshape1 = tf.reshape(pool3, shape=[tf.shape(pool3)[0],3,2])
pool4 = tf.layers.average_pooling1d(inputs=reshape1, pool_size=2, strides=2)
opt = tf.reshape(pool4, shape=[tf.shape(pool4)[0],2])

tensor_opt = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tensor_opt = sess.run(opt, feed_dict={inputs: train_x})

np.savetxt("user_tensor_opt.csv", tensor_opt, delimiter=",")

output = pd.DataFrame(tensor_opt, columns=['tf1','tf2'])
finalDf = pd.concat([output, target], axis = 1)

grpby_finaldf_tf1 = finalDf['tf1'].groupby(finalDf['movie_id'])
grpby_finaldf_tf2 = finalDf['tf2'].groupby(finalDf['movie_id'])
index_list = finalDf.groupby(['movie_id']).size()
finalDf = pd.concat([pd.Series(index_list.index.tolist(), name='movie_id'), grpby_finaldf_tf1.mean(), grpby_finaldf_tf2.mean()], axis=1)
finalDf.to_csv('finalDf.csv')

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('tf1', fontsize = 15)
# ax.set_ylabel('tf2', fontsize = 15)
# ax.set_title('Tf Opt', fontsize = 20)
# targets = list(finalDf['movieId'].tolist())
# cmap = plt.get_cmap('jet')
# colors = cmap(np.linspace(0, 1.0, len(targets)))#['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['movieId'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'tf1']
#                , finalDf.loc[indicesToKeep, 'tf2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()

# plt.show()

