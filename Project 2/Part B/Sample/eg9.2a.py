#
# Chapter 9, Example 2a
#

import numpy as np
import tensorflow as tf
import pylab
import random

import os
if not os.path.isdir('figures'):
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
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# generate training data
seq = []
for _ in range(n_steps):
    seq.append(random.uniform(0,1))
for _ in range(sequence_length):
    seq.append(0.4*seq[-1]*seq[-3] + 0.2*seq[-5] + 0.3*seq[-7] + 0.1*np.random.random())

    
pylab.figure()
pylab.plot(range(len(seq[n_steps:])), seq[n_steps:])
pylab.savefig('./figures/9.2a_1.png')


x_train, y_train = [], []
for i in range(len(seq) - n_steps - 1):
    x_train.append(np.expand_dims(seq[i:i+n_steps], axis=1).tolist())
    y_train.append(np.expand_dims(seq[i+1:i+n_steps+1], axis=1).tolist())

# build the model
x = tf.placeholder(tf.float32,[None, n_steps, n_in])
y = tf.placeholder(tf.float32, [None, n_steps, n_out])
c = tf.placeholder(tf.float32, [None , n_hidden])
h = tf.placeholder(tf.float32, [None , n_hidden])

init_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
                
W = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=1/np.sqrt(n_hidden)))
b = tf.Variable(tf.zeros([n_out]))

cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, x, initial_state = init_state, dtype=tf.float32)

ys = []
for i, o in enumerate(tf.split(outputs, n_steps, axis = 1)):
    y_ = tf.matmul(tf.squeeze(o, [1]), W) + b
    ys.append(y_)

ys_ = tf.stack(ys, axis=1)
cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - ys_), axis=2))
train_op = tf.train.AdamOptimizer(lr).minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

c0, h0 = np.zeros([len(x_train), n_hidden]), np.zeros([len(x_train), n_hidden])
loss = []
for i in range(n_iters):

    loss_, _ = sess.run([cost, train_op], {x:x_train, y: y_train, c: c0, h: h0})
    loss.append(loss_)

    if not i % 100:
        print('iter:%d, cost: %g'%(i, loss[i]))

pylab.figure()
pylab.plot(range(n_iters), loss)
pylab.xlabel('epochs')
pylab.ylabel('mean square error')
pylab.savefig('./figures/9.2a_2.png')


c0, h0 = np.zeros([1, n_hidden]), np.zeros([1, n_hidden])
pred = []
for i in range(len(x_train)):
    state, pred_ = sess.run([states, ys_], {x: [x_train[i]], c: c0, h: h0})
    pred.append(pred_[0, n_steps-1])
    
pylab.figure()
pylab.plot(range(len(seq[n_steps:])), seq[n_steps:])
pylab.plot(range(len(pred)), pred)
pylab.savefig('./figures/9.2a_3.png')


pylab.show()

        




