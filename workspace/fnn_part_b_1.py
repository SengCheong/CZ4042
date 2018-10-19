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

def evaluate_fnn_param(params):

    #============================== PROJECT PARAM STARTS HERE ===============================

    #feature constants
    NUM_FEATURES = 8

    #network parameters - these are the parameters we need to plot for the project
    hidden_neurons = 30
    decay = 0.001
    batch_size = 32

    #training parameters
    learning_rate = 0.0000001
    epochs = params
    ratio = 0.7

    #randomness initialization
    seed = 10
    np.random.seed(seed)

    #============================== PROJECT PARAM ENDS HERE ===================================

    #============================== DATA PROCESSING STARTS HERE ====================================
    #read and divide data into test and train sets 
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')

    #extract the features and targets
    X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]

    # ok so what is going on here is that there are N number of elements originally,  
    # because of the way numpy arrays slice negative indexes, what's going is this
    # cal_housing[:-1] returns an array of the elements taken from the 1st dimension. this means that it is an array of N elements.
    # the tensor expects N by 1 number of elements i.e. 1D tensor or matrix. a 0D tensor is an array or 1-d vector, because it is N elements * kth neurons
    # so we turn it into a matrix so it becomes a matrix of 1 by N elements
    # you can ask, why didn't we do cal_housing[:,-2:-1] cause bitch the numpy is fked. if i is starting index and j is ending index then numpy computes it as n - i and n -j
    # so what happens if we got a 5 by 5 and do [:,-2:-1] it will return all 5 rows, and the columns of (5-2)th index to give a 5 by 1 matrix
    # then you ask, if this is the case then we can't we do [:,-1,0] cause 0 is not negative, so it goes to the start. so how the hell you expect np to iterate from -1 to 0 in a forward manner
    Y_data = (np.asmatrix(Y_data)).transpose()

    #normalize using mean and sd
    X_data = (X_data - np.mean(X_data, axis=0))/ np.std(X_data, axis=0)
    #Y_data = (Y_data - np.mean(Y_data, axis=0))/ np.std(Y_data, axis=0)

    #get the indexes as a list
    idx = np.arange(X_data.shape[0])
    #shuffle the list
    np.random.shuffle(idx)
    #update the dataset
    X_data, Y_data = X_data[idx], Y_data[idx]

    #he split his data here into test and training sets
    partition = X_data.shape[0]//10
    m = partition * 7
    trainX, trainY = X_data[:m], Y_data[:m]
    testX, testY = X_data[m:], Y_data[m:]

    #normalize inputs by using standard normal distribuition
    #trainX = (trainX - np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
    #trainY = (trainX - np.mean(trainX, axis=0))/ np.std(trainX, axis=0)


    #============================== DATA PROCESSING ENDS HERE ====================================

    # ========================== TENSORFLOW TRAINING SHIT STARTS HERE =================================================

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
    # since this is a 1-dimension output, we only got 1 dimension
    weights_o = tf.Variable(tf.truncated_normal([hidden_neurons, 1],stddev=1.0 / np.sqrt(float(hidden_neurons))),name='weights')
    biases_o = tf.Variable(tf.zeros([1]),name='biases')
    y = tf.matmul(hidden, weights_o) + biases_o
    # ~~~~~~~~~~~~~~~~~~ end of output layer ~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~ learning section ~~~~~~~~~~~~~~~~~~~~~~~
    regularization = tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_o) 
    #returns sse
    #given that y_-y returns a rank 2 tensor e.g batch size * 1 
    #reduce_sum with axis = 1 sums all elements in each pattern, then reduce the dimension by 1
    #meaning this becomes a rank 1 tensor, a vector of 1 element
    #reduce_mean will reduce it to a scalar and since there is only 1 element, there is no change 
    #ALWAYS REMEMBER, REDUCE WORKS WITH EACH AXIS'S ELEMENTS. SO WHAT YOU DO IS FROM THE OUTERMOST ELEMENT, ACCESS THE INNER ELEMENT 
    #SO FOR AXIS 0 IT TENSOR-SUMS EACH ELEMENT IN AXIS 0
    #FOR AXIS 1 IT TENSOR-SUMS EACH ELEMENT IN AXIS 
    #this whole op returns the sum as 
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y) + beta*regularization, axis=1))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    #~~~~~~~~~~~~~~~~~~~ end of learning section ~~~~~~~~~~~~~~~~~~~~~~~

    # ========================== TENSORFLOW TRAINING SHIT ENDS HERE =================================================



    # ========================== TENSORFLOW STATISTIC OPERATIONS STARTS HERE ========================================
    #linear output is mean square error
    error = tf.reduce_mean(tf.square(y_ - y))
    sampled_output = tf.squeeze(y)
    # ========================== TENSORFLOW STATISTIC OEPRATIONS END HERE ===========================================


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #timer vars
        time_taken = 0
        total_time_taken = 0

        test_indexes = np.random.choice(range(testY.shape[0]),5,False)

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
                #remember i said we had endpoints? 1 was to feed the computational graph. the other was to access the computed outputs for target learning
                #also note: np arrays take from start to end - 1, so no worries of overlap
                #train_op.run(feed_dict={x: randomized_X[start:end], y_: randomized_Y[start:end], beta: decay})
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], beta: decay})

            end_time = timer()
            time_taken = time_taken + (end_time-start_time)
            total_time_taken = total_time_taken + (end_time-start_time)


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

    epochs = 100
    samples = 5

    result = evaluate_fnn_param(epochs)
    
    #params = ["Train Error: {}".format(i) for i in params]

    plt.figure(1)
    plt.plot(range(epochs), result[1])
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Train Error')

    plt.figure(2)
    plt.plot(range(samples), result[3], 'bo')
    plt.plot(range(samples), result[2], 'ro')
    plt.xlabel(str(samples) + ' Outputs')
    plt.ylabel(str(samples) + ' Targets')
    plt.legend(['Predicted','Actual'])
    plt.show()


if __name__ == '__main__':
    main()