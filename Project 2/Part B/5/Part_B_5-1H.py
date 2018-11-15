import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from google.colab import drive
from timeit import default_timer as timer
from mpl_toolkits import mplot3d

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

batch_size = 128
epochs = 10
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def configure_char_cnn_model(x, keep_chance):

  with tf.device('/gpu:0'):

    input_layer = tf.reshape(tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu,
        name="conv1")
    
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME',
        name="pool1")

    dropout_1 = tf.nn.dropout(pool1, keep_prob=keep_chance)

    conv2 = tf.layers.conv2d(
        dropout_1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu,
        name="conv2")
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME',
        name="pool2")

    dropout_2 = tf.nn.dropout(pool2, keep_prob=keep_chance)

    pool2 = tf.squeeze(tf.reduce_max(dropout_2, 1), squeeze_dims=[1])

    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None,name="logits")

  return logits


def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('/content/gdrive/My Drive/Colab Notebooks/train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('/content/gdrive/My Drive/Colab Notebooks/test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test

def configure_statistics(logits, y_):
    
    # ========================== TENSORFLOW STATISTIC OPERATIONS STARTS HERE ========================================
    # a tensor is produced containing a 1-hot of patterns that were correctly classified and not; logits have been transformed by prev operation into probabilities
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    # average accuracy over the supplied test patterns
    accuracy = tf.reduce_mean(correct_prediction)
    # classification errors over the supplied test patterns
    classification_errors = tf.count_nonzero(tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1)))
    # ========================== TENSORFLOW STATISTIC OPERATION END HERE ===========================================
    
    return correct_prediction, accuracy, classification_errors
    
  
def main():
  tf.reset_default_graph()
  
  drive.mount('/content/gdrive')

  x_train, y_train, x_test, y_test = read_data_chars()

  print(len(x_train))
  print(len(x_test))


  N = len(x_train)
  indexes = np.arange(N)

  #====================== PROJECT STATISTICS START HERE ===============================
  matrix_test_acc_pts = []
  matrix_training_acc_pts = []
  matrix_training_cost_pts = []
  #====================== PROJEC

  for p in range(1,10):
    
    tf.reset_default_graph()
  
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
  
    prob = p/10

    with tf.Session() as sess:

      logits = configure_char_cnn_model(x,prob)
      correct_prediction, accuracy, classification_errors = configure_statistics(logits, tf.one_hot(y_, MAX_LABEL))

      # Optimizer
      entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
      train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
      
      sess.run(tf.global_variables_initializer())
      
      # training
      test_acc_pts = []
      training_acc_pts = []
      training_cost_pts = []
      epoch_times = []


      randomizedX, randomizedY = x_train,y_train

      total_start = timer()

      for e in range(epochs):

          np.random.shuffle(indexes)            
          randomizedX, randomizedY = randomizedX[indexes], randomizedY[indexes]

          experiment_start = timer()
          
          for start, end in zip(range(0, N+1, batch_size), range(batch_size, N+1, batch_size)):
              sess.run([train_op], {x: randomizedX[start:end], y_: randomizedY[start:end]})
              
          experiment_end = timer()

          #upon completing an epoch of training, collect required stats
          loss_pt = entropy.eval(feed_dict={x: randomizedX, y_: randomizedY})
          training_cost_pts.append(loss_pt)
          test_acc_pt = accuracy.eval(feed_dict={x: x_test, y_: y_test})
          test_acc_pts.append(test_acc_pt)
          training_acc_pt = accuracy.eval(feed_dict={x: x_train, y_: y_train})
          training_acc_pts.append(training_acc_pt)
          epoch_times.append(experiment_end-experiment_start)
          
          if(e % 100 == 0):
            print('epoch', e, 'entropy', loss_pt, 'time', experiment_end - experiment_start)

      total_end = timer()

      matrix_test_acc_pts.append(test_acc_pts)
      matrix_training_cost_pts.append(training_cost_pts)
      matrix_training_acc_pts.append(training_acc_pts)

      np_test_accs = np.array(test_acc_pts)
      np_test_accs = np.expand_dims(np_test_accs,axis=0)
      np_training_accs = np.array(training_acc_pts)
      np_training_accs = np.expand_dims(np_training_accs,axis=0)
      np_training_costs = np.array(training_cost_pts)
      np_training_costs = np.expand_dims(np_training_costs,axis=0)
      np_times = np.expand_dims((total_end - total_start, np.mean(epoch_times)),axis=0)

          
      try:
          prev_data_test_accs = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/b1h_test_accs.txt',ndmin=2)
          np_test_accs = np.append(prev_data_test_accs,np_test_accs,axis=0)

          prev_data_training_cost = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/b1h_training_cost.txt',ndmin=2)
          np_training_costs = np.append(prev_data_training_cost,np_training_costs,axis=0)

          prev_data_training_accs = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/b1h_training_acc.txt',ndmin=2)
          np_training_accs = np.append(prev_data_training_accs,np_training_accs,axis=0)

          prev_times = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/b1h_times.txt',ndmin=2)
          np_times = np.append(prev_times, np_times)

          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b1h_training_cost.txt',np_training_costs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b1h_test_accs.txt',np_test_accs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b1h_training_acc.txt',np_training_accs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b1h_times.txt',np_times)
          
      except Exception:
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b1h_training_cost.txt',np_training_costs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b1h_test_accs.txt',np_test_accs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b1h_training_acc.txt',np_training_accs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b1h_times.txt',np_times)
      
      print('Time taken', total_end - total_start)

      sess.close()
      
  legends = ["{} keep-prob".format(p/10) for p in range(1,10)]

  plt.figure(1)
  for acc in matrix_test_acc_pts:
      plt.plot(range(epochs), acc)
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('Test Accuracy')
  plt.legend(legends)

  plt.figure(2)
  for acc in matrix_training_cost_pts:
      plt.plot(range(epochs), acc)
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('Cost')
  plt.legend(legends)

  plt.figure(3)
  for acc in matrix_training_acc_pts:
      plt.plot(range(epochs), acc)
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('Cost')
  plt.legend(legends)

  plt.show()


if __name__ == '__main__':
  main()
