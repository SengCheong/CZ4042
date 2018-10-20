import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import multiprocessing as mp

# scale data by performing vector arithmetic

#============================== PROJECT PARAM STARTS HERE ===============================

#feature constants
NUM_FEATURES = 36
NUM_CLASSES = 6

#network parameters - these are the parameters we need to plot for the project
hidden_neurons = 10
decay = 0.000001
batch_size = 32

#training parameters
learning_rate = 0.01
epochs = 1000

#randomness initialization
seed = 10
np.random.seed(seed)

#============================== PROJECT PARAM ENDS HERE ===================================

def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def evaluate_fnn_param(param):


    #============================== DATA PROCESSING STARTS HERE ====================================
    train_input = np.loadtxt('sat_train.txt',delimiter=' ')

    #separate targets and training data
    trainX = train_input[:,:36]
    train_Y = train_input[:,-1].astype(int)

    #normalize the inputs
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

    #convert class 7 to 6
    train_Y[train_Y == 7] = 6

    #initialize a matrix of zeros with row = number of inputs and col = num of class for 1 hot matrix
    trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))

    #activate 1-hot values
    trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TEST DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #same stuff with test data, so will cut short the coments
    test_input = np.loadtxt('sat_test.txt', delimiter=' ')

    testX = test_input[:,:36]
    test_Y = test_input[:,-1].astype(int)

    testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))

    test_Y[test_Y == 7] = 6
    testY = np.zeros((test_Y.shape[0], NUM_CLASSES))

    testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix

    #============================== DATA PROCESSING ENDS HERE ====================================

    # ========================== TENSORFLOW TRAINING  STARTS HERE ===============================================

    # create tensorflow endpoints for feeding data into the computational graph
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    beta = tf.placeholder(tf.float32)

    # Build the graph for the deep net

    # ~~~~~~~~~~~~~~~~~~ Hidden layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Hidden Layer, Perceptron
    weights_h1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden_neurons],stddev=1.0 / np.sqrt(float(NUM_FEATURES))),name='weights')
    biases_h1 = tf.Variable(tf.zeros([hidden_neurons]),name='biases')
    hidden = tf.nn.sigmoid(tf.matmul(x, weights_h1) + biases_h1)
    # ~~~~~~~~~~~~~~~~~~ end of hidden layer ~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~ output layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output Layer, Linear. the classification is performed outside of the network.
    weights_o = tf.Variable(tf.truncated_normal([hidden_neurons, NUM_CLASSES],stddev=1.0 / np.sqrt(float(hidden_neurons))),name='weights')
    biases_o = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
    logits = tf.matmul(hidden, weights_o) + biases_o
    # ~~~~~~~~~~~~~~~~~~ end of output layer ~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~ learning section ~~~~~~~~~~~~~~~~~~~~~~~
    #perform backprog and classification
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)

    #compute L2 regularization
    regularization = tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_o) 

    #calculate log loss
    loss = tf.reduce_mean(cross_entropy + beta*regularization)

    #a tf variable to keep track of the number of batches fed
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step to keep track of batches
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    # ~~~~~~~~~~~~~~~~~ end of learning section ~~~~~~~~~~~~~~~~~~~

    # ========================== TENSORFLOW TRAINING ENDS HERE =================================================

    # ========================== TENSORFLOW STATISTIC OPERATIONS STARTS HERE ========================================
 
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    classification_errors = tf.count_nonzero(tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1)))
 
    # ========================== TENSORFLOW STATISTIC OEPRATIONS END HERE ===========================================

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #just a list to track each epoch's accuracy
        test_acc = []
        test_log = []
        train_classification = []
        train_log = []

        n = trainX.shape[0]

        #number of training patterns used
        indexes = np.arange(n)

        #timer vars
        time_taken = 0
        total_time_taken = 0
        #iterate over each epoch
        for i in range(epochs):

            #shuffle the indexes
            np.random.shuffle(indexes)

            #build a new list based on random indexes
            randomized_X = trainX[indexes]
            randomized_Y = trainY[indexes]
            start_time = timer()

            #iterate over a range to build the start and end indexes
            for start, end in zip(range(0, n+1, batch_size), range(batch_size, n+1, batch_size)): 
                #train the optimizer; feed the data
                train_op.run(feed_dict={x: randomized_X[start:end], y_: randomized_Y[start:end], beta: decay})

            end_time = timer()
            time_taken = time_taken + (end_time-start_time)
            total_time_taken = total_time_taken + (end_time-start_time)

            #tests 
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY, beta: decay}))
            test_log.append(loss.eval(feed_dict={x: testX, y_: testY, beta: decay}))
            train_classification.append(classification_errors.eval(feed_dict={x: trainX, y_: trainY, beta: decay}))
            train_log.append(loss.eval(feed_dict={x: trainX, y_: trainY, beta: decay}))

            if i % 100 == 0:
                print('iter %d: accuracy -  %g, time taken - %g'%(i, test_acc[i],time_taken))
                time_taken = 0

    batch_time = (total_time_taken/epochs)/(n/batch_size)
    return (param,test_acc,test_log,train_classification, train_log, batch_time)

def main():

    no_threads = mp.cpu_count()

    null_list = [None]

    params = null_list
    p = mp.Pool(processes = no_threads)
    results = p.map(evaluate_fnn_param, params)

    test_accs = []
    test_logs = []
    train_classifications = []
    train_logs = []
    time_taken = []
    
    for result in results:
        test_accs.append(result[1])
        test_logs.append(result[2])
        train_classifications.append(result[3])
        train_logs.append(result[4])
        time_taken.append((result[5]))

    plt.figure(1)
    for acc in test_accs:
        plt.plot(range(epochs), acc)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy against Test Data')
    plt.legend(params)

    plt.figure(2)
    for classification in train_classifications:
        plt.plot(range(epochs), classification)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('classification errors against training data')
    plt.legend(params)

    plt.figure(3)
    for log in test_logs:
         plt.plot(range(epochs), log)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Log Loss against test data')
    plt.legend(params)
    
    plt.figure(4)
    for log in train_logs:
         plt.plot(range(epochs), log)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Log Loss against training')
    plt.legend(params)

    plt.show()

if __name__ == '__main__':
    main()

