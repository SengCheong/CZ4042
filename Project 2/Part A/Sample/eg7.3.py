#
# Chapter 7, Example 3
#


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import pylab

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    
FLAGS = None

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

no_epochs = 100
batch_size = 128

def cnn(x):

  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  u_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
  h_conv1 = tf.nn.relu(u_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
  

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  u_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
  h_conv2 = tf.nn.relu(u_conv2)

  # Second pooling layer.
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return W_conv1, h_conv1, h_pool1, h_conv2, h_pool2, y_conv, keep_prob


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main():
  # Import data
  mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
  trainX, trainY  = mnist.train.images[:12000], mnist.train.labels[:12000]
  testX, testY = mnist.test.images[:2000], mnist.test.labels[:2000]

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  W_conv1, h_conv1, h_pool1, h_conv2, h_pool2, y_conv, keep_prob = cnn(x)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                          logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  N = len(trainX)
  idx = np.arange(N)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    test_acc = []
    for i in range(no_epochs):
      np.random.shuffle(idx)
      trainX, trainY = trainX[idx], trainY[idx]

      for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.5})
      
      test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
      print('iter %d: test accuracy %g'%(i, test_acc[i]))

    pylab.figure()
    pylab.plot(np.arange(no_epochs), test_acc, label='gradient descent')
    pylab.xlabel('epochs')
    pylab.ylabel('test accuracy')
    pylab.legend(loc='lower right')
    pylab.savefig('./figures/7.3_1.png')

    W_conv1_ = sess.run(W_conv1)
    W_conv1_ = np.array(W_conv1_)
    pylab.figure()
    pylab.gray()
    for i in range(32):
      pylab.subplot(8, 4, i+1); pylab.axis('off'); pylab.imshow(W_conv1_[:,:,0,i])
    pylab.savefig('./figures/7.3_2.png')

    ind = np.random.randint(low=0, high=55000)
    X = mnist.train.images[ind,:]

    pylab.figure()
    pylab.gray()
    pylab.axis('off'); pylab.imshow(X.reshape(28,28))
    pylab.savefig('./figures/7.3_3.png')

    h_conv1_, h_pool1_, h_conv2_, h_pool2_ = sess.run([h_conv1, h_pool1, h_conv2, h_pool2],
                                                  {x: X.reshape(1,784)})
    pylab.figure()
    pylab.gray()
    h_conv1_ = np.array(h_conv1_)
    for i in range(32):
        pylab.subplot(4, 8, i+1); pylab.axis('off'); pylab.imshow(h_conv1_[0,:,:,i])
    pylab.savefig('./figures/7.3_4.png')

    pylab.figure()
    pylab.gray()
    h_pool1_ = np.array(h_pool1_)
    for i in range(32):
        pylab.subplot(4, 8, i+1); pylab.axis('off'); pylab.imshow(h_pool1_[0,:,:,i])
    pylab.savefig('./figures/7.3_5.png')
    
    pylab.figure()
    pylab.gray()
    h_conv2_ = np.array(h_conv2_)
    for i in range(64):
        pylab.subplot(8, 8, i+1); pylab.axis('off'); pylab.imshow(h_conv2_[0,:,:,i])
    pylab.savefig('figures/7.3_6.png')

    pylab.figure()
    pylab.gray()
    h_pool2_ = np.array(h_pool2_)
    for i in range(64):
        pylab.subplot(8, 8, i+1); pylab.axis('off'); pylab.imshow(h_pool2_[0,:,:,i])
    pylab.savefig('./figures/7.3_7.png')

    pylab.show()


if __name__ == '__main__':
  main()
