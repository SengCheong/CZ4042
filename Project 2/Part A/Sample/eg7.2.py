#
# Chapter 7, example 2
#


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

seed = 10
np.random.seed(seed)

# read MNIST data and select a random sample
mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
trainX = mnist.train.images
ind = np.random.randint(low=0, high=55000)
X = trainX[ind, :].astype(np.float32)

# filters
W = np.array([[[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]],
     [[1, 2, 1],[0, 0, 0], [-1, -2, -1]],
     [[3, 4, 3], [4, 5, 4], [3, 4, 3]]]).astype(np.float32)

# computational graph
x = tf.placeholder(tf.float32, [1, 28, 28, 1])

w = tf.Variable(W.reshape(3, 3, 1, 3), tf.float32)
b = tf.Variable(np.zeros(3, dtype=np.float32), tf.float32)


u = tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME') + b
y = tf.nn.sigmoid(u)
o = tf.nn.avg_pool(y, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

# initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# evaluate
u_, y_, o_ = sess.run([u, y, o], {x: X.reshape([1, 28, 28, 1])})

print(u_.shape)

# plot input image
plt.figure()
plt.gray()
plt.axis('off')
plt.imshow(X.reshape(28,28))
plt.savefig('./figures/7.2_1.png')

# plot u
plt.figure()
plt.gray()
plt.subplot(3,1,1), plt.axis('off'), plt.imshow(u_[0,:,:,0])
plt.subplot(3,1,2), plt.axis('off'), plt.imshow(u_[0,:,:,1])
plt.subplot(3,1,3), plt.axis('off'), plt.imshow(u_[0,:,:,2])
plt.savefig('./figures/7.2_2.png')

# plot output y of convolution layer
plt.figure()
plt.gray()
plt.subplot(3,1,1), plt.axis('off'), plt.imshow(y_[0,:,:,0])
plt.subplot(3,1,2), plt.axis('off'), plt.imshow(y_[0,:,:,1])
plt.subplot(3,1,3), plt.axis('off'), plt.imshow(y_[0,:,:,2])
plt.savefig('./figures/7.2_3.png')

# plot output o of pooling layer
plt.figure()
plt.gray()
plt.subplot(3,1,1), plt.axis('off'), plt.imshow(o_[0,:,:,0])
plt.subplot(3,1,2), plt.axis('off'), plt.imshow(o_[0,:,:,1])
plt.subplot(3,1,3), plt.axis('off'), plt.imshow(o_[0,:,:,2])
plt.savefig('./figures/7.2_4.png')

plt.show()

