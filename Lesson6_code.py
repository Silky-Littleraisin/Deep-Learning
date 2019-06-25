import pandas as pd
import tensorflow as tf
import numpy as np

from pprint import pprint
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
        tf.test
    ]

except AttributeError:
    print('Unrequired modules are already deleted (Skipped).')


def load_mnist():
    # 学習データ
    x_train = np.load('/Users/silky/Downloads/chap06/data/x_train.npy')
    t_train = np.load('/Users/silky/Downloads/chap06/data/t_train.npy')

    # テストデータ
    x_test = np.load('/Users/silky/Downloads/chap06/data/x_test.npy')

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    t_train = np.eye(10)[t_train.astype('int32').flatten()]

    return (x_train, x_test, t_train)

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(1234)
random_state = 42

### レイヤー定義 ###
#
class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1, 1, 1, 1], padding='VALID'):
        # Heの初期値
        fan_in = np.prod(filter_shape[:3])  #
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
            low=-np.sqrt(6 / fan_in),
            high=np.sqrt(6 / fan_in),
            size=filter_shape
        ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')  #
        self.function = function
        self.strides = strides
        self.padding = padding

    def __call__(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)


class Pooling:
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def __call__(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)


class Flatten:
    def __call__(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))


class Dense:
        def __init__(self, in_dim, out_dim, function=lambda x: x):
            # He Initialization
            self.W = tf.Variable(rng.uniform(
                low=-np.sqrt(6 / in_dim),
                high=np.sqrt(6 / in_dim),
                size=(in_dim, out_dim)
            ).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
            self.function = function

        def __call__(self, x):
            return self.function(tf.matmul(x, self.W) + self.b)


def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))
    
### ネットワーク ###
import matplotlib.pyplot as plt

x_train, x_test, t_train = load_mnist()
print(len(x_train),len(x_test),len(t_train))
print(x_test[0])

#print(B.shape)
# for i in range(20):
#     X=np.array(x_test[i]).reshape(28,28)
#     plt.imshow(X, cmap="gray")
#     plt.show()


x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.1, random_state=random_state)
print(len(x_train),len(x_valid),len(t_train),len(t_valid))

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])

h = Conv((5, 5, 1, 20), tf.nn.relu)(x)           # 28x28x 1 -> 24x24x20
h = Pooling((1, 2, 2, 1))(h)                           # 24x24x20 -> 12x12x20
h = Conv((4, 4, 20, 50), tf.nn.relu)(h)        # 12xx20 ->  9xx50
#h = Pooling((1, 2, 2, 1))(h)
h = Conv((2, 2, 50, 50), tf.nn.relu)(h)       # 9 ->8
h = Pooling((1, 2, 2, 1))(h)                #  8x x50 ->  4x x50
h = Flatten()(h)
y = Dense(4*4*50, 10, tf.nn.softmax)(h)
#y = Dense(50, 10, tf.nn.softmax)(h)

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

### 学習 ###

n_epochs = 109
batch_size = 100
n_batches = x_train.shape[0]//batch_size

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y_pred2 = sess.run(y, feed_dict={x: x_test})
    print(y_pred2.argmax(axis=1)[:20], len(y_pred2.argmax(axis=1)))
    submission = pd.Series(y_pred2.argmax(axis=1), name='label')
    submission.to_csv('/Users/silky/Downloads/chap06/submission_pred0.csv', header=True, index_label='id')
    for epoch in range(n_epochs):
        x_train, t_train = shuffle(x_train, t_train, random_state=random_state)
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: x_train[start:end], t: t_train[start:end]})
        y_pred, cost_valid = sess.run([y, cost], feed_dict={x: x_valid, t: t_valid})
        print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
            epoch,
            cost_valid,
            accuracy_score(t_valid.argmax(axis=1), y_pred.argmax(axis=1))
        ))
        if accuracy_score(t_valid.argmax(axis=1), y_pred.argmax(axis=1))>=0.93:
            break
    y_pred2 = sess.run(y, feed_dict={x: x_test})
for i in range(20):
    X=np.array(x_test[i]).reshape(28,28)
    plt.imshow(X, cmap="gray")
    plt.title(y_pred2.argmax(axis=1)[i])
    # グラフを表示
    plt.show()

print(y_pred2.argmax(axis=1)[:20],len(y_pred2.argmax(axis=1)))
submission = pd.Series(y_pred2.argmax(axis=1), name='label')
submission.to_csv('/Users/silky/Downloads/chap06/submission_pred.csv', header=True, index_label='id')
