
import tensorflow as tf
import csv

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(1234)

### define layers ###
tf.reset_default_graph()
z_dim = 10

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

def encoder(x):
    # WRITE ME

def sampling_z(mean, var):
    # WRITE ME

def decoder(z):
    # WRITE ME

def lower_bound(x):
    #Encode
    mean, var = encoder(x)
    KL = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + tf_log(var) - mean**2 - var, axis=1))
    
    #Z
    z = sampling_z(mean, var)
    
    #Decode
    y = decoder(z)
    reconstruction = tf.reduce_mean(tf.reduce_sum(x * tf_log(y) + (1 - x) * tf_log(1 - y), axis=1))
    
    lower_bound = [-KL, reconstruction]
    
    return lower_bound


### training ###
#学習データと検証データに分割
x_train, x_valid = train_test_split(x_train, test_size=0.1)

x = tf.placeholder(tf.float32, [None, 784])
lower_bound = lower_bound(x)

cost = -tf.reduce_sum(lower_bound)
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(cost)

valid = tf.reduce_sum(lower_bound)

batch_size =100

n_batches = x_train.shape[0] // batch_size
n_epochs = 5

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(n_epochs):
    # WRITE ME
    
    
### sampling ###
x = tf.placeholder(tf.float32, [None, 784])
sample_z_func = encoder(x)

z = tf.placeholder(tf.float32, [None, z_dim])
sample_x_func = decoder(z)

# Encode
mean, var = sess.run(sample_z_func, feed_dict={x: x_test})
sample_z = mean

# Decode
sample_x = sess.run(sample_x_func, feed_dict={z: sample_z})


### to_csv ###
with open('/root/userspace/chap10/materiaks/sample_submission.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(sample_x.reshape(-1, 28*28).tolist())