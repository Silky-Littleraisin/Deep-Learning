import os

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    del [
        tf.app,
        tf.compat,
        tf.contrib,
        tf.estimator,
        tf.gfile,
        tf.graph_util,
        tf.image,
        tf.initializers,
        tf.keras,
        tf.layers,
        tf.logging,
        tf.losses,
        tf.metrics,
        tf.python_io,
        tf.resource_loader,
        tf.saved_model,
        tf.sets,
        tf.summary,
        tf.sysconfig,
        tf.test,
        tf.train
    ]
except AttributeError:
    print('Unrequired modules are already deleted (Skipped).')


def load_fashionmnist():
    # 学習データ
    x_train = np.load('/Users/silky/Downloads/chap05/data/x_train.npy')
    y_train = np.load('/Users/silky/Downloads/chap05/data/y_train.npy')

    # テストデータ
    x_test = np.load('/Users/silky/Downloads/chap05/data/x_test.npy')

    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32')]
    x_test = x_test.reshape(-1, 784).astype('float32') / 255

    return x_train, y_train, x_test
import math
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

tf.reset_default_graph() # グラフのリセット

x_train, t_train, x_test = load_fashionmnist()

x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=10000)

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool) # 訓練時orテスト時

class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.W = tf.Variable(tf.random_uniform(shape=(in_dim, out_dim), minval=-0.08, maxval=0.08), name='W')
        self.b = tf.Variable(tf.zeros(out_dim), name='b')
        self.function = function

        self.params = [self.W, self.b]

    def __call__(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)

def sgd(cost, params, eta=0.01):
    grads = tf.gradients(cost, params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append(param.assign_sub(eta * grad))
    return updates

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

layers = [
    Dense(784, 200, tf.nn.relu),
    Dense(200, 200, tf.nn.relu),
    Dense(200, 10, tf.nn.softmax)
]

params = []
h = x
for layer in layers:
    h = layer(h)
    params += layer.params
y = h

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))

updates = sgd(cost, params)
train = tf.group(*updates)

n_epochs = 50
batch_size = 100
n_batches = math.ceil(len(x_train) / batch_size)
print('x_train',len(x_train))
print('n_batches',n_batches)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
    x_train, t_train = shuffle(x_train, t_train)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(train, feed_dict={x: x_train[start:end], t: t_train[start:end], is_training: True})
    y_pred, cost_valid_ = sess.run([y, cost], feed_dict={x: x_valid, t: t_valid, is_training: False})
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
        epoch + 1,
        cost_valid_,
        accuracy_score(t_valid.argmax(axis=1), y_pred.argmax(axis=1))
    ))
    y_pred2=sess.run(y,feed_dict={x:x_test})
    # if epoch==0:
    #     y_preds=y_pred.argmax(axis=1)
    y_preds=y_pred2.argmax(axis=1)

    # if epoch!=0:
    #     y_preds=np.concatenate((y_preds,y_pred.argmax(axis=1)),axis=None)
    print(np.shape(y_pred))
    print(np.shape(y_preds))
    if accuracy_score(t_valid.argmax(axis=1), y_pred.argmax(axis=1))>=0.9:
        break
    # WRITE ME

# y_preds=y_preds.argmax(axis=1)
submission = pd.Series(y_preds, name='label')
submission.to_csv('/Users/silky/Downloads/chap05/materials/submission_pred.csv', header=True, index_label='id')
