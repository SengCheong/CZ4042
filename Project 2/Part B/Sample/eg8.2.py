#
# Chapter 8, Example 2
#


import numpy as np
import tensorflow as tf


n_hidden = 3
n_in = 2
n_out = 1
n_steps = 3
n_seq = 2


X1 = np.array([[1, 2], [-1, 1], [0, 3]])
X2 = np.array([[-1, 0], [2, -1], [3, -1]])


W = tf.Variable(np.array([[2.0, 1.3, -1.0]]), dtype = tf.float64)
b = tf.Variable(np.array([0.2, 0.2, 0.2]), dtype = tf.float64)
U = tf.Variable(np.array([[-1.0, 0.5, 0.2], [0.5, 0.1, -2.0]]), dtype = tf.float64)
V = tf.Variable(np.array([[2.0], [-1.5], [0.2]]), dtype = tf.float64)
c = tf.Variable(np.array([0.1]), dtype=tf.float64)


x = tf.placeholder(tf.float64, [n_seq, n_in])
init_state = tf.placeholder(tf.float64, [n_seq, n_out])


z = tf.matmul(x, U) + tf.matmul(init_state, W) + b
h = tf.tanh(z)
u = tf.matmul(h, V) + c
y = tf.sigmoid(u)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

y_ = np.zeros([n_seq, n_out])
for t in range(n_steps):
    h_, y_ = sess.run([h, y], {x: np.transpose([X1[t], X2[t]]), init_state: y_ })
    print('h: {}'.format(h_))
    print('y: {}'.format(y_))

         
            


