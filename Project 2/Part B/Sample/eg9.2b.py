#
# Chapter 9, Example 2b
#

import numpy as np
import tensorflow as tf
import pylab
import random

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

n_in = 1
n_hidden = 12
n_out = 1
n_steps = 8
n_seqs = 1
sequence_length = 256

n_iters = 2000
lr = 0.001

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

# generate training data
seq = []
for i in range(n_steps):
    seq.append(random.uniform(0, 1))
for i in range(sequence_length):
    seq.append(0.4*seq[-1]*seq[-3] + 0.2*seq[-5] + 0.3*seq[-7] + 0.1*random.random())

x_train, y_train = [], []
for i in range(len(seq) - n_steps - 1):
    x_train.append(np.expand_dims(seq[i:i+n_steps], axis=1).tolist())
    y_train.append(np.expand_dims(seq[i+1:i+n_steps+1], axis=1).tolist())

# build the model
x = tf.placeholder(tf.float32,[None, n_steps, n_in])
y = tf.placeholder(tf.float32, [None, n_steps, n_out])
init_state = tf.placeholder(tf.float32, [2, None, n_hidden])
                
W = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=1/np.sqrt(n_hidden)))
b = tf.Variable(tf.zeros([n_out]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, reuse=tf.get_variable_scope().reuse)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, reuse=tf.get_variable_scope().reuse)
cells = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)

ys = []
for i, h in enumerate(tf.split(outputs, n_steps, axis = 1)):
    y_ = tf.matmul(tf.squeeze(h, [1]), W) + b
    ys.append(y_)

ys_ = tf.stack(ys, axis=1)
cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - ys_), axis=2))
train_op = tf.train.AdamOptimizer(lr).minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

state = np.zeros([2, len(x_train), n_hidden])
loss = []
for i in range(n_iters):

    sess.run(train_op, {x:x_train, y: y_train, init_state: state})
    loss.append(sess.run(cost, {x:x_train, y: y_train, init_state: state}))

    if not i % 100:
        print('iter:%d, cost: %g'%(i, loss[i]))

pylab.figure()
pylab.plot(range(n_iters), loss)
pylab.xlabel('epochs')
pylab.ylabel('mean square error')
pylab.savefig('./figures/9.2b_1.png')

state = np.zeros([2, 1, n_hidden])
pred = []
for i in range(len(x_train)):
    state, pred_ = sess.run([states, ys_], {x: [x_train[i]]})
    pred.append(pred_[0, n_steps-1])
    
pylab.figure()
pylab.plot(range(len(seq[n_steps:])), seq[n_steps:])
pylab.plot(range(len(pred)), pred)
pylab.savefig('./figures/9.2b_2.png')


pylab.show()

        




