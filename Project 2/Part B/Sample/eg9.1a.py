#
# Chapter 9, Example 1a
#

import numpy as np
import tensorflow as tf
import pylab

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

n_in = 2
n_hidden = 10
n_out = 1
n_steps = 16
n_seqs = 64

n_iters = 10000
lr = 0.001

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


# generate training data
x_train = np.random.rand(n_seqs*n_steps, n_in)
y_train = np.zeros([n_seqs*n_steps, n_out])

y_train[7:,0] = 5*x_train[6:-1, 0]*x_train[5:-2,1] - 2*x_train[:-7,0] + 3.5*x_train[2:-5,1]**2
y_train += 0.1*np.random.randn(n_seqs*n_steps, n_out)

x_train = x_train.reshape(n_seqs, n_steps, n_in)
y_train = y_train.reshape(n_seqs, n_steps, n_out)


# build the model
x = tf.placeholder(tf.float32,[None, n_steps, n_in])
y = tf.placeholder(tf.float32, [None, n_steps, n_out])
init_state = tf.placeholder(tf.float32, [None, n_hidden])
                
W = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=1/np.sqrt(n_hidden)))
b = tf.Variable(tf.zeros([n_out]))

cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

ys = []
for i, h in enumerate(tf.split(outputs, n_steps, axis = 1)):
    y_ = tf.matmul(tf.squeeze(h), W) + b
    ys.append(y_)

ys_ = tf.stack(ys, axis=1)
cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - ys_), axis=2))
train_op = tf.train.AdamOptimizer(lr).minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

state = np.zeros([n_seqs, n_hidden])
loss = []
for i in range(n_iters):

    sess.run(train_op, {x:x_train, y: y_train, init_state: state})
    loss.append(sess.run(cost, {x:x_train, y: y_train, init_state: state}))

    if not i % 100:
        print('iter:%d, cost: %g'%(i, loss[i]))

pred = sess.run(ys_, {x:x_train, y: y_train, init_state: state})

pylab.figure()
pylab.plot(range(n_iters), loss)
pylab.xlabel('epochs')
pylab.ylabel('mean square error')
pylab.savefig('./figures/9.1a_1.png')

pylab.figure(figsize=(4, 2))
for i in range(8):
    ax = pylab.subplot(4, 2, i+1)
    pylab.axis('off')
    ax.plot(range(n_steps), x_train[i,:,0],'r')
    ax.plot(range(n_steps), x_train[i,:,1],'g')
    pylab.savefig('./figures/9.1a_2.png')
    
pylab.figure(figsize=(4, 2))
for i in range(8):
    ax = pylab.subplot(4, 2, i+1)
    pylab.axis('off')
    ax.plot(range(n_steps), y_train[i,:,0])
    pylab.savefig('./figures/9.1a_3.png')

pylab.figure(figsize=(4, 2))
for i in range(8):
    ax = pylab.subplot(4, 2, i+1)
    pylab.axis('off')
    ax.plot(range(n_steps), y_train[i,:,0])
    ax.plot(range(n_steps), pred[i, :, 0])
    pylab.savefig('./figures/9.1a_4.png')


pylab.show()

        




