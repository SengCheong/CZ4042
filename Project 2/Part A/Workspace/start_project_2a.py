
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
epochs = 10
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#helper method. don't need to know what is going on and don't want to know. it works
def load_data(file):

    with open('My Drive/Colab Notebooks/{}'.format(file), 'rb') as fo:
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
        #create the bias for 50 feature maps
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

        #for some reason, the dimensionas are flattened. it's possible softmax can oly process 2D tensors, hence the flattening
        dim = pool_1.get_shape()[1].value * pool_1.get_shape()[2].value * pool_1.get_shape()[3].value 
        #thus, they flatten here 
        pool_1_flat = tf.reshape(pool_1, [-1, dim])
        
        #Softmax weights and bias; 
        #for a normal softmax, tf.truncated_normal([hidden_neurons, NUM_CLASSES]) returns a tensor whose row is the number of hidden layer neurons, and col is kth softmax neuron's weight to the output of the hidden layer neuron 
        #in the CNN, it's a tensor whose row is the flattened input, and the column is the kth softmax's weight 
        W2 = tf.Variable(tf.truncated_normal([dim, NUM_CLASSES], stddev=1.0/np.sqrt(dim)), name='weights_3')
        b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_3')
        logits = tf.matmul(pool_1_flat, W2) + b2

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

    #use tensorflow's way of updating the weights
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    #create the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #run the optimizer
    train_op = optimizer.minimize(loss)


    # ========================== TENSORFLOW STATISTIC OPERATIONS STARTS HERE ========================================
    #note, tensorflow operations are loaded and executed per epoch

    #tf.argmax(logits,1) behaves like np.max, with logits axis = 1. probably axis = 0 is the number of input patterns. so returns the index that has highest probability
    #so tf.argmax(logits,1) will return the index of the kth neuron
    #remember y_ is a 1-hot, so tf.argmax will return the index with the 1-hot value
    #however, not sure if it does the evaluation over the entire dataset supplied or per input pattern. seems like entire dataset judging from next satement
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    #probably average accuracy per pattern; when they reduce means they do map_reduce, not literally reduce
    accuracy = tf.reduce_mean(correct_prediction)
    classification_errors = tf.count_nonzero(tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1)))
    # ========================== TENSORFLOW STATISTIC OEPRATIONS END HERE ===========================================

    #number of input patterns, each pattern is RGB image of 32 by 32 by 3 channels flattened into a single array of 3072 points
    N = len(trainX)
    indexes = np.arange(N)
    randomizedX, randomizedY = trainX, trainY

    start = timer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(indexes)
            

            randomizedX, randomizedY = randomizedX[idx], trainY[idx]

            #one way to receive endpoint
            #_, loss_val = sess.run([train_op, loss], {x: randomizedX, y_: randomizedY})
            #another way
            #train_op.run(feed_dict={x: randomized_X[start:end], y_: randomized_Y[start:end]})


            print('epoch', e, 'entropy', loss_)

    
    end = timer()
    
    print("Time Taken: {}".format(end-start))
    
    plt.figure()
    plt.gray()
    #originally, the image was a flattened 1-d tensor; first we reshape it back to its original dimensions.
    #however to show it we need to reorder the arrangements to height, width and channels. 
    #so originally NUM_CHANNELS was axis 0; height was axis 1, width was axis 2l
    #tranpose(1,2,0) reorders them as such to show the image proper
    ind = np.random.randint(low=0, high=10000)
    X = trainX[ind,:]
    X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    plt.axis('off')
    plt.imshow(X_show)


if __name__ == '__main__':
  main()
