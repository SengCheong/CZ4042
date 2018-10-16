#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
#use matplotlib because of fucked up
#import pylab as plt
import matplotlib.pyplot as plt

# scale data by performing vector arithmetic
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

#feature constants,
NUM_FEATURES = 36
NUM_CLASSES = 6

#network parameters
learning_rate = 0.01
num_neurons = 10

#training parameters
epochs = 1000
batch_size = 32

#random weight initialization 
seed = 10
np.random.seed(seed)

#read training data, space delimited.
train_input = np.loadtxt('sat_train.txt',delimiter=' ')

#for all rows, extract the first 36 columns - corresponds to the 36 features
trainX = train_input[:,:36]
#for all rows, extract the last element. numpy interprets reverse index as n - 1
train_Y = train_input[:,-1].astype(int)

#axis=0 refers to the 0th dimension vector with the lowest elements 
#for a n-dimension vector, axis=k where k belongs in n, min/max access iterates over each element in 0 to k-1 dimensions, 
#and uses the kth dimension to compare, returning a ndarray of min elements from the kth dimension over 0 to k-1 dimensions
#so for a 5 by 5 matrix with numbers 1 to 25, np.min(matrix, axis=0 returns the row/k=0 vector, which 1,2,3,45
#for axis 1, goes over each row, then for a 

#for the project, dimension 0 is the number of input patterns, which is a single input vector
#normalize the input features 
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

#convert class 7 to 6
train_Y[train_Y == 7] = 6

#initialize a matrix of zeros with row = number of inputs and col = num of class
trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))

#convert the zero matrix to a 1-hot by iterating using numpy arrays as index, a feature of numpy
#uses index array, DOES NOT ITERATE over the array
#basically, array indexing occurs when an array is supplied to index of the manipulated array
#what happens is it uses the supplied stuf to compute a list of array indexs, extract the elements of these computed indexes from the 
#manipulate array, then returns a new array with dimensions based on the number of supplied ranegs
#e.g. a[2:5, np.array([2,4]) ] returns a new array of 3 (2:5) by 2 (2:4) with the object elements being
#a[2-5,2] and a[2-5, 4]
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

# experiment with small datasets, there are over 2000 patterns
#get the first 1000 patterns of 36 input features from trainX, and trainY for the training data labels
trainX = trainX[:1000]
trainY = trainY[:1000]

#count the number of input patterns as n
n = trainX.shape[0]

# Create the model
# tf.placeholder creates an placeholder of a tensor as an endpoint for feeding data or retrieving data from the NN
# the first paramter is the data type, the second is an Iterable of N dimensions as the shape of the tensor, with None as any number
# hence x refers to the training dataset endpoint for tf and so forth for y
# this is required becuase tf propagates the number of training patterns applied -> think mini-batch SGD and BGD
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Build the graph for the deep net
	
#truncated normal distribuition is used as the weights initialization
weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')

#the synaptic input. note matmul vs element wise multiplication. no idea atm if it's the entire batch's synaptic or single neuron synaptic
logits  = tf.matmul(x, weights) + biases

#compute the cross entropy, recall that for softmax, delta_U involves K and U, where K is the one_hot matrix described by y_, and logits 
#as the synaptic input vector. returns a tensor; starting to think this returns the cross entropy for each neuron?
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)

#returns average loss across all patterns in a epoch ..? think because this is batch, so the average loss per neuron needs to be computed?
loss = tf.reduce_mean(cross_entropy)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#probably the part where weights and bias are updated, presented as a computational graph internally
train_op = optimizer.minimize(loss)

#tf.argmax(logits,1) behaves like np.max, with logits axis = 1. probably axis = 0 is the number of input patterns
#so if that's the case tf.argmax(y_,1) extracts the target for the current pattern?
correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
#probably average accuracy per epoch?
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #just a list to track each epoch's accuracy
    train_acc = []

    #iterate over each epoch
    for i in range(epochs):

    	#remember i said we had endpoints? 1 was to feed the computational graph. the other was to access the computed outputs for
    	#classication checking
        train_op.run(feed_dict={x: trainX, y_: trainY})
        train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))

        if i % 100 == 0:
            print('iter %d: accuracy %g'%(i, train_acc[i]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()

