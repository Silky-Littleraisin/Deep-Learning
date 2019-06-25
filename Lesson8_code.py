import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

import numpy as np
from pprint import pprint


def load_dataset():
    # 学習データ
    x_train = np.load('/Users/silky/Downloads/chap08/data/x_train.npy')
    t_train = np.load('/Users/silky/Downloads/chap08/data/t_train.npy')

    # テストデータ
    x_test = np.load('/Users/silky/Downloads/chap08/data/x_test.npy')

    return (x_train, x_test, t_train)


x_train, x_test, t_train = load_dataset()


### レイヤー定義 ###
class Embedding:
    def __init__(self, vocab_size, emb_dim, scale=0.08):
        self.V = tf.Variable(tf.random_normal([vocab_size, emb_dim], stddev=scale), name='V')

    def __call__(self, x):
        return tf.nn.embedding_lookup(self.V, x)


class RNN:
    def __init__(self, hid_dim, seq_len=None, initial_state=None):
        self.cell = tf.nn.rnn_cell.BasicRNNCell(hid_dim)
        self.initial_state = initial_state
        self.seq_len = seq_len

    def __call__(self, x):
        if self.initial_state is None:
            self.initial_state = self.cell.zero_state(tf.shape(x)[0], tf.float32)

        outputs, state = tf.nn.dynamic_rnn(self.cell, x, self.seq_len, self.initial_state)
        return tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(x)[0]), self.seq_len - 1], axis=1))

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))


import tensorflow.contrib.eager as tfe


class EagerEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_dim, scale=0.08):
        super(EagerEmbedding, self).__init__()

        self.V = self.add_variable("V", [vocab_size, emb_dim], initializer='RandomNormal')

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.V, inputs)


class EagerRNN(tf.keras.layers.Layer):
    def __init__(self, hid_dim):
        super(EagerRNN, self).__init__()

        self.hid_dim = hid_dim

    def build(self, input_shape):
        self.W = self.add_variable("W", [input_shape[-1] + self.hid_dim, self.hid_dim], initializer='Orthogonal')
        self.b = self.add_variable("b", [self.hid_dim], initializer='Zeros')

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.hid_dim))
        for t in range(inputs.shape[1]):
            state = tf.nn.tanh(tf.matmul(tf.concat([inputs[:, t, :], state], axis=1), self.W) + self.b)
            output = state
            outputs.append(output)
        return outputs[-1]


class LSTM:
    def __init__(self, hid_dim, seq_len=None, initial_state=None):
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(hid_dim)
        self.initial_state = initial_state
        self.seq_len = seq_len

    def __call__(self, x):
        if self.initial_state is None:
            self.initial_state = self.cell.zero_state(tf.shape(x)[0], tf.float32)

        outputs, state = tf.nn.dynamic_rnn(self.cell, x, self.seq_len, self.initial_state)
        return tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(x)[0]), self.seq_len - 1], axis=1))


class Model(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, hid_dim, scale=0.08):
        super(Model, self).__init__()

        self.word_embedding = EagerEmbedding(vocab_size, emb_dim)
        self.rnn = EagerRNN(hid_dim)
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        h = self.word_embedding(inputs)
        h = self.rnn(h)
        y = self.dense(h)
        return tf.reshape(y, shape=[-1])


# loss関数
def loss_with_logits(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32)))

# tf.enable_eager_execution()

### グラフ構築 ###
tf.reset_default_graph()

# emb_dim = 1
# hid_dim = 1
a=1
if a==1:
    num_words = max([max(s) for s in np.hstack((x_train, x_test))])
    pad_index = 0
    #
    # x = tf.placeholder(tf.int32, [None, None], name='x')
    # t = tf.placeholder(tf.float32, [None, None], name='t')
    #
    # seq_len = tf.reduce_sum(tf.cast(tf.not_equal(x, pad_index), tf.int32), axis=1)
    #
    # pprint(seq_len)
    #
    # h = Embedding(num_words, emb_dim)(x)
    # h = LSTM(emb_dim, hid_dim, seq_len)(h)
    #
    # # h = RNN(hid_dim, seq_len)(h)
    # y = tf.layers.Dense(1, tf.nn.sigmoid)(h)
    #
    # cost = -tf.reduce_mean(t * tf_log(y) + (1 - t) * tf_log(1 - y))
    #
    # train = tf.train.AdamOptimizer().minimize(cost)
    # test = tf.round(y)
    tf.reset_default_graph() # グラフ初期化

    emb_dim = 100
    hid_dim = 50

    x = tf.placeholder(tf.int32, [None, None], name='x')
    t = tf.placeholder(tf.float32, [None, None], name='t')

    seq_len = tf.reduce_sum(tf.cast(tf.not_equal(x, pad_index), tf.int32), axis=1)
    pprint(seq_len)

    h = Embedding(num_words, emb_dim)(x)
    h = LSTM(hid_dim, seq_len)(h)
    y = tf.layers.Dense(1, tf.nn.sigmoid)(h)

    cost = -tf.reduce_mean(t*tf_log(y) + (1 - t)*tf_log(1 - y))

    train = tf.train.AdamOptimizer().minimize(cost)
    test = tf.round(y)

    ### データの準備 ###
    x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train)

    ### 学習 ###
    n_epochs = 50
    batch_size = 100
    n_batches_train = len(x_train) // batch_size
    n_batches_valid = len(x_valid) // batch_size
    n_batches_test = len(x_test) // batch_size


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(1)
        for epoch in range(n_epochs):
            # Train
            train_costs = []
            for i in range(n_batches_train):
                start = i * batch_size
                end = start + batch_size

                x_train_batch = np.array(pad_sequences(x_train[start:end], padding='post', value=pad_index))
                t_train_batch = np.array(t_train[start:end])[:, None]

                _, train_cost = sess.run([train, cost], feed_dict={x: x_train_batch, t: t_train_batch})
                train_costs.append(train_cost)

            # Valid
            print(2)
            valid_costs = []
            y_pred = []
            y_pred2=[]
            for i in range(n_batches_valid):
                start = i * batch_size
                end = start + batch_size

                x_valid_pad = np.array(pad_sequences(x_valid[start:end], padding='post', value=pad_index))
                t_valid_pad = np.array(t_valid[start:end])[:, None]
                pred, valid_cost = sess.run([test, cost], feed_dict={x: x_valid_pad, t: t_valid_pad})

                y_pred += pred.flatten().tolist()
                valid_costs.append(valid_cost)

            for i in range(n_batches_test):
                x_test_pad = np.array(pad_sequences(x_test[start:end], padding='post', value=pad_index))

                pred2 = sess.run(test, feed_dict={x: x_test_pad})
                y_pred2 += pred2.flatten().tolist()

            print(3)
            submission = pd.Series(y_pred2, name='label')
            submission.to_csv('/Users/silky/Downloads/chap08/materials/submission_pred.csv', header=True, index_label='id')

            print('EPOCH: %i, Training Cost: %.3f, Validation Cost: %.3f, Validation F1: %.3f' % (
                epoch + 1, np.mean(train_costs), np.mean(valid_costs), f1_score(t_valid, y_pred, average='macro')))
else:
    emb_dim = 100
    hid_dim = 50
    num_words = max([max(s) for s in np.hstack((x_train, x_test))])
    x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train)

    pad_index = 0
    num_words = 10000
    x_train = pad_sequences(x_train, padding='post', value=pad_index)
    x_valid = pad_sequences(x_valid, padding='post', value=pad_index)

    ### 学習 ###
    n_epochs = 5
    batch_size = 100

    model = Model(num_words, emb_dim, hid_dim)

    optimizer = tf.train.AdamOptimizer()

    epoch_num = 10
    train_dataset = [(c,b) for c,b in zip(x_train, t_train)]
    # train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    for epoch in range(epoch_num):
        tf.train.get_or_create_global_step()

        for (i, (inputs, labels)) in enumerate(train_dataset):
            optimizer.minimize(lambda: loss_with_logits(model(inputs), labels),
                               global_step=tf.train.get_global_step())

        valid_logits = model(x_test)
        valid_loss = loss_with_logits(valid_logits, t_valid)
        valid_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(tf.nn.sigmoid(valid_logits)), t_valid), tf.float32))
        y_pred2 += valid_logits.flatten().tolist()

        print(valid_logits,('EPOCH %02d\t Validation Loss: %.2f\t Validation Accuracy: %.2f') % (epoch + 1, valid_loss, valid_accuracy))

        submission = pd.Series(y_pred2, name='label')
        submission.to_csv('/Users/silky/Downloads/chap08/materials/submission_pred.csv', header=True, index_label='id')
