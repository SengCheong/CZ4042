#
# Project 1, starter code part b
#

import math
import tensorflow as tf
import numpy as np
#use matplotlib because of fucked up
#import pylab as plt
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import multiprocessing as mp



#============================== PROJECT PARAM STARTS HERE ===============================

#feature constants
NUM_FEATURES = 8

#network parameters - these are the parameters we need to plot for the project
hidden_neurons = 30
decay = 0.001
batch_size = 32

#training parameters
learning_rate = 0.0000001
epochs = 10

#randomness initialization
seed = 10
np.random.seed(seed)

samples = 5
#============================== PROJECT PARAM ENDS HERE ===================================



def evaluate_fnn_param(params):

    #============================== DATA PROCESSING STARTS HERE ====================================
    #read and divide data into test and train sets 
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')

    #extract the features and targets
    X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]

    #transpose the matrix because of np reading oddity
    Y_data = (np.asmatrix(Y_data)).transpose()

    #normalize using mean and sd
    #X_data = (X_data - np.mean(X_data, axis=0))/ np.std(X_data, axis=0)
    #Y_data = (Y_data - np.mean(Y_data, axis=0))/ np.std(Y_data, axis=0)
    
    #get the indexes as a list
    idx = np.arange(X_data.shape[0])
    #shuffle the list
    np.random.shuffle(idx)
    #randomize the dataset
    X_data, Y_data = X_data[idx], Y_data[idx]

    #we split his data here into test and training sets
    partition = X_data.shape[0]//10
    m = partition * 7
    trainX, trainY = X_data[:m], Y_data[:m]
    testX, testY = X_data[m:], Y_data[m:]

    trainX = (trainX - np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
    testX = (testX - np.mean(testX, axis=0))/ np.std(testX, axis=0)

    #============================== DATA PROCESSING ENDS HERE ====================================

    # ========================== TENSORFLOW TRAINING  STARTS HERE =================================================

    # Create the model endpoints for feeding data and comparision
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, 1])
    beta = tf.placeholder(tf.float32)

    # ~~~~~~~~~~~~~~~~~~ Hidden layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Hidden Layer, relu
    weights_h1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden_neurons],stddev=1.0 / np.sqrt(float(NUM_FEATURES))),name='weights')
    biases_h1 = tf.Variable(tf.zeros([hidden_neurons]),name='biases')
    hidden = tf.nn.relu(tf.matmul(x, weights_h1) + biases_h1)
    # ~~~~~~~~~~~~~~~~~~ end of hidden layer ~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~ output layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # since this is a 1-dimension output, we only got 1 dimension represented by 1 neuron
    weights_o = tf.Variable(tf.truncated_normal([hidden_neurons, 1],stddev=1.0 / np.sqrt(float(hidden_neurons))),name='weights')
    biases_o = tf.Variable(tf.zeros([1]),name='biases')
    y = tf.matmul(hidden, weights_o) + biases_o
    # ~~~~~~~~~~~~~~~~~~ end of output layer ~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~ learning section ~~~~~~~~~~~~~~~~~~~~~~~
    regularization = tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_o) 
    #reduce_sum sums the square error for each pattern, effectively performing sum of square errors for batch gradient descent
    #the tensor shape at before summing is matris with batch size * 1
    #the tensor shape at after summing is a vector of 1 element
    #reduce_mean reduces the tensor into a scalar 
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y) + beta*regularization, axis=1))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    #~~~~~~~~~~~~~~~~~~~ end of learning section ~~~~~~~~~~~~~~~~~~~~~~~

    # ========================== TENSORFLOW TRAINING  ENDS HERE =================================================

    # ========================== TENSORFLOW STATISTIC OPERATIONS STARTS HERE ========================================
    #linear neuron error is mean square error.
    #the tensor shape at before meaning is matris with batch size * 1
    #the tensor shape at after meaning is a vector of 1 element. eval will turn it into a scalar
    error = tf.reduce_mean(tf.square(y_ - y))

    summed = tf.reduce_sum(tf.square(y_ - y))

    squared = tf.Print(summed,[summed, tf.shape(tf.square(y_-y))],"Summed")
    meaned = tf.Print(error,[error],"Meaned")
    #the tensor shape at after 

    #sampled_output is for retreiving the outputs for 50 samples, squeeze here compress the outputs from a matrix into vector
    sampled_output = tf.squeeze(y)
    # ========================== TENSORFLOW STATISTIC OEPRATIONS END HERE ===========================================


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #timer vars
        time_taken = 0
        total_time_taken = 0

        test_indexes = np.random.choice(range(testY.shape[0]),samples,False)

        training_err = []
        testing_err = []

        n = trainX.shape[0]
        indexes = np.arange(n)

        for i in range(epochs):

            np.random.shuffle(indexes)

            randomized_X = trainX[indexes]
            randomized_Y = trainY[indexes]

            start_time = timer()

            for start, end in zip(range(0, n+1, batch_size), range(batch_size, n+1, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], beta: decay})

            end_time = timer()
            time_taken = time_taken + (end_time-start_time)
            total_time_taken = total_time_taken + (end_time-start_time)

            print("BREAK")
            print(squared.eval(feed_dict={x: trainX, y_: trainY, beta: decay}))
            print(meaned.eval(feed_dict={x: trainX, y_: trainY, beta: decay}))

            train_err = error.eval(feed_dict={x: trainX, y_: trainY, beta: decay})
            training_err.append(train_err)

            test_err = error.eval(feed_dict={x: testX, y_: testY, beta: decay})
            testing_err.append(test_err)

            if i % 100 == 0:
                print('iter %d: test error %g'%(i, training_err[i]))

        outputs = sampled_output.eval(feed_dict={x: testX[test_indexes], y_: testY[test_indexes], beta: decay})
        targets = testY[test_indexes]

    print("Total Time Taken: {}".format(total_time_taken))

    return (training_err,testing_err,outputs,targets)

def main():


    result = evaluate_fnn_param(epochs)
    

    plt.figure(1)
    plt.plot(range(epochs), result[0])
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Training Error')

    plt.figure(2)
    plt.plot(range(epochs), result[1])
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test Error')


    plt.figure(3)
    plt.plot(range(samples), result[3], 'bo')
    plt.plot(range(samples), result[2], 'ro')
    plt.xlabel(str(samples) + ' Outputs')
    plt.ylabel(str(samples) + ' Targets')
    plt.legend(['Predicted','Actual'])
    plt.show()


if __name__ == '__main__':
    main()