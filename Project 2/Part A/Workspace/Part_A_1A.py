
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import multiprocessing as mp
from google.colab import drive


NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 3000
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

# for allowing gpu use
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#helper method. don't need to know what is going on and don't want to know. it works
def load_data(file):

    with open('/content/gdrive/My Drive/Colab Notebooks/{}'.format(file), 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    #one-hot vector vector
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_


def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


def cnn(images):

    #use gpu
    with tf.device('/gpu:0'):
        #reshape the flattened array back to the original image dimension + 1; the -1 allows the tensorflow graph to be feed any number of images
        images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
        
        #===initialization of weights and bias for convolutional layer 1===
        # - create a truncated_normal initalized weights of the following dimensions: 9 by 9 kernel by 3 channels for 50 weights/feature_maps/filters
        W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
        b1 = tf.Variable(tf.zeros([50]), name='biases_1')
        #===creation of actual convolutional area convolutional layer by initializing the activation function====
        # tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') creates the actual convolutional tensor computation
        # it's parameters are the input, the weights, The stride of the sliding window for each *dimension* of input, the padding
        # as there are 4 paramters corresponding to batch index, IMG_SIZE,IMG_SIZE,NUM_CHANNELS
        conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
        #ksize refers to kernel size over batch index,pooling window size, pooling window size, channel
        #we put channel = 1 so that the window works only on a single channel at a time
        #for strides, the stride is taken over batch index, feature_map_horizontal, feature_map_vertical, channel
        #meaning it strides over 1 channel, at 2 by 2, 1 data input at a time
        pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')


        #== convolutional layer 2
        W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0/np.sqrt(50*5*5)), name='weights_2')
        b2 = tf.Variable(tf.zeros([60]), name='biases_2')
        conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
        pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

        #for some reason, the dimensionas are flattened. it's possible normal neurons can only process 2D tensors, hence the flattening
        dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
        pool_2_flat = tf.reshape(pool_2, [-1, dim])
        

        W3 = tf.Variable(tf.truncated_normal([dim,300], stddev=1.0/np.sqrt(dim)), name='weights_3')
        b3 = tf.Variable(tf.zeros([300]), name='biases_3')
        hidden_3 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)

        #Softmax weights and bias; 
        #for a normal softmax, tf.truncated_normal([hidden_neurons, NUM_CLASSES]) returns a tensor whose row is the number of hidden layer neurons, and col is kth softmax neuron's weight to the output of the hidden layer neuron 
        #in the CNN, it's a tensor whose row is the flattened input, and the column is the kth softmax's weight 
        W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_4')
        b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
        logits = tf.matmul(hidden_3, W4) + b4

    return logits


def main():
    drive.mount('/content/gdrive')
  
    #just know there was some voodoo magic and boom your data is arranged nice and fine
    #note, the image data is a flattened tensor
    trainX, trainY = load_data('training_data')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_data')
    print(testX.shape, testY.shape)

    #standard normal distribution
    #trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

    #scale the data, as per Question 1
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

    # Create the model endpoints for feeding input data = x and the training data = y_
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    
    #create the cnn and get the synaptic input of the softmax layer
    logits = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    # ========================== TENSORFLOW STATISTIC OPERATIONS STARTS HERE ========================================
    # a tensor is produced containing a 1-hot of patterns that were correctly classified and not; logits have been transformed by prev operation into probabilities
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    # average accuracy over the supplied test patterns
    accuracy = tf.reduce_mean(correct_prediction)
    
    # classification errors over the supplied test patterns
    classification_errors = tf.count_nonzero(tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1)))
    # ========================== TENSORFLOW STATISTIC OPERATION END HERE ===========================================
    
    
    #number of input patterns, each pattern is RGB image of 32 by 32 by 3 channels flattened into a single array of 3072 points
    N = len(trainX)
    indexes = np.arange(N)
    randomizedX, randomizedY = trainX, trainY

    
    #====================== PROJECT STATISTICS START HERE ===============================
    test_acc_pts = []
    training_cost_pts = []
    #====================== PROJECT STATISTICS START HERE ===============================
    start = timer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(indexes)            
            randomizedX, randomizedY = randomizedX[indexes], randomizedY[indexes]
            
            for start, end in zip(range(0, N+1, batch_size), range(batch_size, N+1, batch_size)):
                sess.run([train_op], {x: randomizedX[start:end], y_: randomizedY[start:end]})
                
            #upon completing an epoch of training, collect required stats
            loss_pt = loss.eval(feed_dict={x: randomizedX, y_: randomizedY})
            training_cost_pts.append(loss_pt)
            test_acc_pt = accuracy.eval(feed_dict={x: testX, y_: testY})
            test_acc_pts.append(test_acc_pt)
 
            
            if(e % 100 == 0):
              print('epoch', e, 'entropy', loss_pt)
    
    
    plt.figure(1)
    plt.plot(range(epochs), test_acc_pts)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy against Test Data')
    

    plt.figure(2)
    plt.plot(range(epochs), training_cost_pts)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Training Cost')

    plt.show()


if __name__ == '__main__':
  main()
