import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import csv

import numpy as np
import pickle


def pickle_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_data():
    # 学習データ
    x_train = pickle_load('/Users/silky/Downloads/chap09/data/x_train.pkl')
    t_train = pickle_load('/Users/silky/Downloads/chap09/data/t_train.pkl')
    tokenizer_en = np.load('/Users/silky/Downloads/chap09/data/tokenizer_en.npy').item()
    tokenizer_ja = np.load('/Users/silky/Downloads/chap09/data/tokenizer_ja.npy').item()



    # テストデータ
    x_test = pickle_load('/Users/silky/Downloads/chap09/data/x_test.pkl')

    return (x_train, t_train, tokenizer_en, tokenizer_ja, x_test)


x_train, t_train, tokenizer_en, tokenizer_ja, x_test = load_data()

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

### レイヤー定義 ###
class Embedding:
    def __init__(self, vocab_size, emb_dim, scale=0.08):
        self.V = tf.Variable(tf.random_normal([vocab_size, emb_dim], stddev=scale), name='V')

    def __call__(self, x):
        return tf.nn.embedding_lookup(self.V, x)


class LSTM:
    def __init__(self, hid_dim, seq_len, initial_state, return_state=False, return_sequences=False, hold_state=False,
                 name=None):
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(hid_dim)
        self.initial_state = initial_state
        self.seq_len = seq_len
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.hold_state = hold_state
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name):
            outputs, state = tf.nn.dynamic_rnn(self.cell, x, self.seq_len, self.initial_state)

        if self.hold_state:
            self.initial_state = state

        if not self.return_sequences:
            outputs = state.h

        if not self.return_state:
            return outputs

        return outputs, state

### グラフ構築 ###
# tf.reset_default_graph()
#
# emb_dim = 1
# hid_dim = 1
pad_index = 0

x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train)


tf.reset_default_graph() # グラフ初期化

emb_dim = 256
hid_dim = 256

x = tf.placeholder(tf.int32, [None, None], name='x')
seq_len = tf.reduce_sum(tf.cast(tf.not_equal(x, pad_index), tf.int32), axis=1)
t = tf.placeholder(tf.int32, [None, None], name='t')
seq_len_t_in = tf.reduce_sum(tf.cast(tf.not_equal(t, pad_index), tf.int32), axis=1) - 1

t_out = tf.one_hot(t[:, 1:], depth=ja_vocab_size, dtype=tf.float32)
t_out = t_out * tf.expand_dims(tf.cast(tf.not_equal(t[:, 1:], pad_index), tf.float32), axis=-1)

initial_state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([tf.shape(x)[0], hid_dim]), tf.zeros([tf.shape(x)[0], hid_dim]))

# Encoder
h_e = Embedding(en_vocab_size, emb_dim)(x)
_, encoded_state = LSTM(hid_dim, seq_len, initial_state, return_state=True, name='encoder_lstm')(h_e)

# Decoder
decoder = [
    Embedding(ja_vocab_size, emb_dim),
    LSTM(hid_dim, seq_len_t_in, encoded_state, return_sequences=True, name='decoder_lstm'),
    tf.layers.Dense(ja_vocab_size, tf.nn.softmax)
]

# Decoderに変数を通す
h_d = decoder[0](t)
h_d = decoder[1](h_d)
y = decoder[2](h_d)

cost = -tf.reduce_mean(tf.reduce_sum(t_out * tf_log(y[:, :-1]), axis=[1, 2]))

train = tf.train.AdamOptimizer().minimize(cost)
#
# x = tf.placeholder(tf.int32, [None, None], name='x')
# t = tf.placeholder(tf.int32, [None, None], name='t')

x_train_lens = [len(com) for com in x_train]
sorted_train_indexes = sorted(range(len(x_train_lens)), key=lambda x: -x_train_lens[x])

x_train = [x_train[ind] for ind in sorted_train_indexes]
t_train = [t_train[ind] for ind in sorted_train_indexes]

n_epochs = 10
batch_size = 128
n_batches = len(x_train) // batch_size

sess = tf.Session()

sess.run(tf.global_variables_initializer())

bos_eos = tf.placeholder(tf.int32, [2], name='bos_eos')
max_len = tf.placeholder(tf.int32, name='max_len')  # iterationの繰り返し回数の限度


def cond(t, continue_flag, init_state, seq_last, seq):
    unfinished = tf.not_equal(tf.reduce_sum(tf.cast(continue_flag, tf.int32)), 0)
    return tf.logical_and(t < max_len, unfinished)


def body(t, prev_continue_flag, init_state, seq_last, seq):
    decoder[1].initial_state = init_state

    # Decoderの再構築
    h = decoder[0](tf.expand_dims(seq_last, axis=-1))
    h = decoder[1](h)
    y = decoder[2](h)

    seq_t = tf.reshape(tf.cast(tf.argmax(y, axis=2), tf.int32), shape=[-1])
    next_state = decoder[1].initial_state

    continue_flag = tf.logical_and(prev_continue_flag, tf.not_equal(seq_t, bos_eos[1]))  #

    return [t + 1, continue_flag, next_state, seq_t, seq.write(t, seq_t)]


decoder[1].hold_state = True
decoder[1].seq_len = None

seq_0 = tf.ones([tf.shape(x)[0]], tf.int32)*bos_eos[0]

t_0 = tf.constant(1)
f_0 = tf.cast(tf.ones_like(seq_0), dtype=tf.bool) #
seq_array = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True).write(0, seq_0)

*_, seq = tf.while_loop(cond, body, loop_vars=[t_0, f_0, encoded_state, seq_0, seq_array])

res = tf.transpose(seq.stack())

bos_id_ja, eos_id_ja = tokenizer_ja.texts_to_sequences(['<s> </s>'])[0]






### 出力 ###
def get_raw_contents(dataset, num, bos_id, eos_id):
    result = []
    for index in dataset[num]:
        if index == eos_id:
            break
            
        result.append(index)
        
        if index == bos_id:
            result = []
            
    return result

for epoch in range(n_epochs):
    # train
    train_costs = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        x_train_batch = np.array(pad_sequences(x_train[start:end], padding='post', value=pad_index))
        t_train_batch = np.array(pad_sequences(t_train[start:end], padding='post', value=pad_index))

        _, train_cost = sess.run([train, cost], feed_dict={x: x_train_batch, t: t_train_batch})
        train_costs.append(train_cost)

    # valid
    x_valid_pad = np.array(pad_sequences(x_valid, padding='post', value=pad_index))
    t_valid_pad = np.array(pad_sequences(t_valid, padding='post', value=pad_index))

    valid_cost = sess.run(cost, feed_dict={x: x_valid_pad, t: t_valid_pad})

    y_pred = sess.run(res, feed_dict={
        x: pad_sequences(x_test, padding='post', value=pad_index),
        bos_eos: np.array([bos_id_ja, eos_id_ja]),
        max_len: 100
    })

    output = [get_raw_contents(y_pred, i, bos_id_ja, eos_id_ja) for i in range(len(y_pred))]

    with open('/Users/silky/Downloads/chap09/materials/submission_gen.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(output)

    print('success_write')
    print('EPOCH: %i, Training Cost: %.3f, Validation Cost: %.3f' % (epoch+1, np.mean(train_costs), valid_cost))



output = [get_raw_contents(y_pred, i, bos_id_ja, eos_id_ja) for i in range(len(y_pred))]

with open('/Users/silky/Downloads/chap09/materials/submission_gen.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(output)
