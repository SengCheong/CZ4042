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

def evaluate_fnn_param(param):

    #============================== PROJECT PARAM STARTS HERE ===============================

    #feature constants
    NUM_FEATURES = 8

    #network parameters - these are the parameters we need to plot for the project
    hidden_neurons = 30
    decay = 0.001
    batch_size = 32

    #training parameters
    learning_rate = param
    epochs = 1000
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
    # ========================== TENSORFLOW STATISTIC OEPRATIONS END HERE ===========================================


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        n = trainX.shape[0]

        #timer vars
        time_taken = 0
        total_time_taken = 0

        folds = 5
        fold_partition_size = n//5

        fold_errs = []
        fold_epoch_errors = []

        for fold in range(folds):

            fold_start = fold * fold_partition_size
            fold_end = (fold+1) * fold_partition_size

            fold_test_X = trainX[fold_start:fold_end]
            fold_test_Y = trainY[fold_start:fold_end]
            fold_train_X = np.append(trainX[:fold_start],trainX[fold_end:], axis=0)
            fold_train_Y = np.append(trainY[:fold_start],trainY[fold_end:], axis=0)

            for i in range(epochs//folds):

                m = fold_train_X.shape[0]
                indexes = np.arange(m)
                np.random.shuffle(indexes)

                randomized_X = fold_train_X[indexes]
                randomized_Y = fold_train_Y[indexes]

                start_time = timer()

                for start, end in zip(range(0, n+1, batch_size), range(batch_size, n+1, batch_size)):
                    train_op.run(feed_dict={x: randomized_X[start:end], y_: randomized_Y[start:end], beta: decay})

                end_time = timer()
                time_taken = time_taken + (end_time-start_time)
                total_time_taken = total_time_taken + (end_time-start_time)

                fold_epoch_error = error.eval(feed_dict={x: fold_test_X, y_: fold_test_Y, beta: decay})
                fold_epoch_errors.append(fold_epoch_error)

                if i % 100 == 0:
                    print('param: %g fold: %d iter %d: epoch error %g'%(param, fold, i, fold_epoch_errors[i]))

            fold_err = error.eval(feed_dict={x: fold_test_X, y_: fold_test_Y, beta: decay})
            fold_errs.append(fold_err)

    #cv error
    fold_training_err = np.array(fold_errs)
    average_test_error = np.mean(fold_training_err)

    print("Total Time Taken: {}".format(total_time_taken))

    return (learning_rate,average_test_error,fold_epoch_errors)

def main():

    no_threads = mp.cpu_count()

    epochs = 1000
    learning_rates = [0.5*10**-6,10**-7,0.5*10**-8,10**-9,10**-10]

    params = learning_rates
    p = mp.Pool(processes = no_threads)
    results = p.map(evaluate_fnn_param, params)
    
    params = ["Learning Rates: {}".format(i) for i in params]

    print(learning_rates)

    cv_errors,test_errors = [],[]
    for result in results:
    	cv_errors.append(result[1])
    	test_errors.append(result[2])

    print(learning_rates)
    print(cv_errors)

    plt.figure(1)
    for rate,cv in zip(learning_rates,cv_errors):
        plt.plot(rate, cv, 'o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Cross Validation Error')
    plt.legend(params)

    plt.figure(2)
    for test_error in test_errors:
        plt.plot(range(epochs), test_error)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Mean Square Error')
    plt.legend(params)
    
    plt.show()


if __name__ == '__main__':
    main()
