#
# Chapter 7, Example 1
#

import tensorflow as tf
import numpy as np
import pylab


# Input image
I = np.array([[0.5, -0.1, 0.2, 0.3, 0.5],
              [0.8, 0.1, -0.5, 0.5, 0.1],
              [-1.0, 0.2, 0.0, 0.3, -0.2],
              [0.7, 0.1, 0.2, -0.6, 0.3],
              [-0.4, 0.0, 0.2, 0.3, -0.3]]).astype(np.float32)

# filter
W = np.array([[0, 1, 1],[1, 0, 1], [1, 1, 0]]).astype(np.float32)

# computational graph
x = tf.placeholder(tf.float32, [1, 5, 5, 1])

w = tf.Variable(W.reshape(3, 3, 1, 1), tf.float32)
b = tf.Variable([0.05], tf.float32)

u = tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'VALID') + b
y = tf.nn.sigmoid(u)

# initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# evaluate u and y
u_, y_ = sess.run([u, y], {x: I.reshape([1, 5, 5, 1])})

print('VALID padding for convolution')
print('u: %s'%u_.reshape([3, 3]))
print('y: %s'%y_.reshape([3, 3]))

# evaluate o for VALID
o = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
o_ = sess.run(o, {x: I.reshape([1, 5, 5, 1])})
print('VALID padding for pooling')
print('o: %s'%o_.reshape([1, 1]))

# evaluate o for SAME 
o = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
o_ = sess.run(o, {x: I.reshape([1, 5, 5, 1])})
print('SAME padding for pooling')
print('o: %s'%o_.reshape([2, 2]))


# evaluate again for SAME padding for convolutions
u = tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME') + b
y = tf.nn.sigmoid(u)
o = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


u_, y_, o_ = sess.run([u, y, o], {x: I.reshape([1, 5, 5, 1])})
print('SAME padding for convolution and pooling')
print('u: %s'%u_.reshape([5, 5]))
print('y: %s'%y_.reshape([5, 5]))
print('o: %s'%o_.reshape([3, 3]))
