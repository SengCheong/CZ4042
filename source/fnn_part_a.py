#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
#use matplotlib because of fucked up
#import pylab as plt
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# scale data by performing vector arithmetic
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

#============================== PROJECT PARAM STARTS HERE ===============================

#feature constants
NUM_FEATURES = 36
NUM_CLASSES = 6

#network parameters - these are the parameters we need to plot for the project
hidden_neurons = 10
decay = 0.000001
batch_size = 64

#training parameters
learning_rate = 0.001
epochs = 1000

#randomness initialization
seed = 10
np.random.seed(seed)

#============================== PROJECT PARAM ENDS HERE ===================================

#============================== DATA PROCESSING STARTS HERE ====================================
#read training data, space delimited.
train_input = np.loadtxt('sat_train.txt',delimiter=' ')

#count the number of input patterns as N
N = train_input.shape[0]

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
#so if we picked 2 iterables, what it is going to do is use the first element of the first iterable, then use the 1st element of the 2nd iterable
#then advance both iterables together. this is what this line of code is doing since both are numpy arrays
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

# experiment with small datasets, there are over 2000 patterns
trainX = trainX[:]
trainY = trainY[:]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TEST DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#same stuff with test data, so will cut short the coments
test_input = np.loadtxt('sat_test.txt', delimiter=' ')

M = train_input.shape[0]

testX = train_input[:,:36]
test_Y = train_input[:,-1].astype(int)

testX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

test_Y[train_Y == 7] = 6
testY = np.zeros((train_Y.shape[0], NUM_CLASSES))

testY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#============================== DATA PROCESSING ENDS HERE ====================================

# ========================== TENSORFLOW TRAINING SHIT STARTS HERE =================================================
# Create the model
# tf.placeholder creates an placeholder of a tensor as an endpoint for feeding data or retrieving data from the NN
# the first paramter is the data type, the second is an Iterable of N dimensions as the shape of the tensor, with None as any number
# hence x refers to the training dataset endpoint for tf and so forth for y
# this is required becuase tf propagates the number of training patterns applied -> think mini-batch SGD and BGD
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Build the graph for the deep net

# ~~~~~~~~~~~~~~~~~~ Hidden layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hidden Layer, Perceptron
  #really the weights. the truncated normal returns a 1-D tensor aka 2d array. whose row is feature n, and col is kth neuron's weight to n
weights_h1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden_neurons],stddev=1.0 / np.sqrt(float(NUM_FEATURES))),name='weights')
#really the bias, tf.zeros returns a n-D tensor, in this case 1 row of 0s whose col is the kth nueron's bias
biases_h1 = tf.Variable(tf.zeros([hidden_neurons]),name='biases')
#activation function. we using perceptron hidden layer
hidden = tf.nn.sigmoid(tf.matmul(x, weights_h1) + biases_h1)
# ~~~~~~~~~~~~~~~~~~ end of hidden layer ~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~ output layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output Layer, Linear. the classification is performed outside of the network.
#really the weights. the truncated normal returns a 1-D tensor aka 2d array. whose row is the number of hidden layer neurons, and col is kth neuron's weight to hidden layer's kth neuron synaptic input
weights_o = tf.Variable(tf.truncated_normal([hidden_neurons, NUM_CLASSES],stddev=1.0 / np.sqrt(float(hidden_neurons))),name='weights')
biases_o = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
#https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits
#so we supply a linear matrix set first, then the tf function deal with it
logits = tf.matmul(hidden, weights_o) + biases_o
# ~~~~~~~~~~~~~~~~~~ end of output layer ~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~ learning section ~~~~~~~~~~~~~~~~~~~~~~~
#compute the cross entropy, recall that for softmax, delta_U involves K and U, where K is the one_hot matrix described by y_, and logits 
#https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits
#so we supply a linear matrix set first, then the tf function deal with the sigmoidal
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)

#idk why he use a placeholder to feed the beta instead of direct assignment, but meh
beta = tf.placeholder(tf.float32)
#returns average loss across all patterns in a epoch ..? think because this is batch, so the average loss per neuron needs to be computed?
regularization = tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_o) #=========================================================================== NEEED THE REGULIZATIONS
loss = tf.reduce_mean(cross_entropy + beta*regularization)

#a tf variable to keep track of the number of batches fed
global_step = tf.Variable(0, name='global_step', trainable=False)
# Use the optimizer to apply the gradients that minimize the loss
# (and also increment the global step counter) as a single training step to keep track of batches
# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)
# ~~~~~~~~~~~~~~~~~ end of learning section ~~~~~~~~~~~~~~~~~~~

# ========================== TENSORFLOW TRAINING SHIT ENDS HERE =================================================

# ========================== TENSORFLOW STATISTIC OPERATIONS STARTS HERE ========================================
#note, tensorflow operations are loaded and executed per epoch

#tf.argmax(logits,1) behaves like np.max, with logits axis = 1. probably axis = 0 is the number of input patterns. so returns the index that has highest probability
#so tf.argmax(logits,1) will return the index of the kth neuron
#remember y_ is a 1-hot, so tf.argmax will return the index with the 1-hot value
#however, not sure if it does the evaluation over the entire dataset supplied or per input pattern. seems like entire dataset judging from next satement
correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
#probably average accuracy per pattern; when they reduce means they do map_reduce, not literally reduce
accuracy = tf.reduce_mean(correct_prediction)

# ========================== TENSORFLOW STATISTIC OEPRATIONS END HERE ===========================================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #just a list to track each epoch's accuracy
    train_acc = []

    n = trainX.shape[0]

    #number of training patterns used
    indexes = np.arange(n)

    #timer vars
    time_taken = 0

    #iterate over each epoch
    for i in range(epochs):

        #shuffle the indexes
        np.random.shuffle(indexes)

        #remember the array indexing of numpy? came back to haunt you. takes the indexes, builds a new array by taking each element from trainX/trainY
        # based on the index returned iterating over indexes
        randomized_X = trainX[indexes]
        randomized_Y = trainY[indexes]

        #zip takes iterators, and compactly combines them into 1 tuple, with each element of the tuple taken from each iterator supplied
        #terminates when any iterator runs out of element to supply
        #so for each epoch, takes the entire data set in mani-atch patterns
        #compute the new weights from the loss of the current data pattern, then updates the weights and bias
        #so eventually for each epoch the entire dataset is consumed
        #we add 1 because range runs from 0 to n-1. numpy array indexing also runs from 0 to n - 1
        #suppose n is 128. then zip will return a tuple (0,8) to (112,120) but not (120,128). the last 8 patterns will be missed

        start_time = timer()

        for start, end in zip(range(0, n+1, batch_size), range(batch_size, n+1, batch_size)):
            #remember i said we had endpoints? 1 was to feed the computational graph. the other was to access the computed outputs for target learning
            #also note: np arrays take from start to end - 1, so no worries of overlap
            train_op.run(feed_dict={x: randomized_X[start:end], y_: randomized_Y[start:end], beta: decay})

        end_time = timer()
        time_taken = time_taken + (end_time-start_time)

        #at the end of each epoch, we do classification errors checking
        #we pass a set of data, then evaluate the logits
        train_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

        if i % 100 == 0:
            print('iter %d: accuracy -  %g, time taken - %g'%(i, train_acc[i],time_taken))
            time_taken = 0


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()
