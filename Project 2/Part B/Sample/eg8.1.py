#
# Chapter 8, Example 1
#


import numpy as np
import tensorflow as tf


n_hidden = 3
n_in = 2
n_out = 2
n_steps = 4
n_seq = 1


X = np.array([[1, 2], [-1, 1], [0, 3], [2, -1]])


W = tf.Variable(np.array([[2.0, 1.3, -1.0], [1.5, 0.0, -0.5 ], [-0.2, 1.5, 0.4]]), dtype = tf.float64)
b = tf.Variable(np.array([0.2, 0.2, 0.2]), dtype = tf.float64)
U = tf.Variable(np.array([[-1.0, 0.5, 0.2], [0.5, 0.1, -2.0]]), dtype = tf.float64)
V = tf.Variable(np.array([[2.0, -1.0], [-1.5, 0.5], [0.2, 0.8]]), dtype = tf.float64)
c = tf.Variable(np.array([0.5, 0.1]), dtype=tf.float64)

x = tf.placeholder(tf.float64, X.shape[1])
init_state = tf.placeholder(tf.float64, [n_hidden])


z = tf.tensordot(tf.transpose(U), x, axes=1) + tf.tensordot(tf.transpose(W), init_state, axes=1) + b
h = tf.tanh(z)
u = tf.tensordot(tf.transpose(V), h, axes=1) + c
y = tf.sigmoid(u)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


h_ = np.array([0, 0, 0])
for t in range(n_steps):
    h_, y_ = sess.run([h, y], {x: X[t], init_state: h_})
    print('h: {}'.format(h_))
    print('y: {}'.format(y_))

         
            


