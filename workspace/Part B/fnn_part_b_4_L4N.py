import math
import tensorflow as tf
import numpy as np
#use matplotlib because of fucked up
#import pylab as plt
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import multiprocessing as mp

def evaluate_fnn_param(params, first_layer_neurons):

    #============================== PROJECT PARAM STARTS HERE ===============================

    #feature constants
    NUM_FEATURES = 8

    #network parameters - these are the parameters we need to plot for the project
    first_neurons = first_layer_neurons
    hidden_neurons = 20
    decay = 0.001
    batch_size = 32

    #training parameters
    learning_rate = 10**-9
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
    weights_h1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, first_neurons],stddev=1.0 / np.sqrt(float(NUM_FEATURES))),name='weights')
    biases_h1 = tf.Variable(tf.zeros([first_neurons]),name='biases')
    hidden_1 = tf.nn.relu(tf.matmul(x, weights_h1) + biases_h1)

    # ~~~~~~~~~~~~~~~~~~ end of hidden layer ~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~ Hidden layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Hidden Layer, relu
    weights_h2 = tf.Variable(tf.truncated_normal([first_neurons, hidden_neurons],stddev=1.0 / np.sqrt(float(NUM_FEATURES))),name='weights')
    biases_h2 = tf.Variable(tf.zeros([hidden_neurons]),name='biases')
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, weights_h2) + biases_h2)
    # ~~~~~~~~~~~~~~~~~~ end of hidden layer ~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~ output layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # since this is a 1-dimension output, we only got 1 dimension
    weights_o = tf.Variable(tf.truncated_normal([hidden_neurons, 1],stddev=1.0 / np.sqrt(float(hidden_neurons))),name='weights')
    biases_o = tf.Variable(tf.zeros([1]),name='biases')
    y = tf.matmul(hidden_2, weights_o) + biases_o
    # ~~~~~~~~~~~~~~~~~~ end of output layer ~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~ learning section ~~~~~~~~~~~~~~~~~~~~~~~
    regularization = tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_o) + tf.nn.l2_loss(weights_h2)
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

    print("Total Time Taken: {}".format(total_time_taken))

    return (training_err,testing_err,False)
    
def main():

    epochs = 100

    result = evaluate_fnn_param(epochs)
    
    #params = ["Train Error: {}".format(i) for i in params]

    plt.figure(1)
    plt.plot(range(epochs), result[1])
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Training Error')

    plt.figure(2)
    plt.plot(range(epochs), result[1])
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test Error')
    plt.show()


if __name__ == '__main__':
    main()