import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from google.colab import drive
from timeit import default_timer as timer
from mpl_toolkits import mplot3d

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

epochs = 2
lr = 0.01
batch_size = 512

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def rnn_model(x,keep_chance):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)

  cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_chance)

  _, encoding = tf.nn.static_rnn(cell_with_dropout, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits

def data_read_words():

  tf.reset_default_graph()
  drive.mount('/content/gdrive')
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('/content/gdrive/My Drive/Colab Notebooks/train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open('/content/gdrive/My Drive/Colab Notebooks/test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words

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
  global n_words

  x_train, y_train, x_test, y_test, n_words = data_read_words()


  N = len(x_train)
  indexes = np.arange(N)

  #====================== PROJECT STATISTICS START HERE ===============================
  matrix_test_acc_pts = []
  matrix_training_acc_pts = []
  matrix_training_cost_pts = []
  #====================== PROJEC


  for p in range(1,10):

    prob = p/10

    tf.reset_default_graph()

    with tf.Session() as sess:
      
      # Create the model
      x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
      y_ = tf.placeholder(tf.int64)


      logits = rnn_model(x,prob)
      correct_prediction, accuracy, classification_errors = configure_statistics(logits, tf.one_hot(y_, MAX_LABEL))

      entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
      train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

      sess.run(tf.global_variables_initializer())
      
      # training
      test_acc_pts = []
      training_cost_pts = []
      training_acc_pts =[]
      epoch_times = []

      randomizedX, randomizedY = x_train,y_train
      testX,testY = x_test,y_test


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

      print('Time taken', total_end - total_start)

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
          prev_data_test_accs = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/b4h_test_accs.txt',ndmin=2)
          np_test_accs = np.append(prev_data_test_accs,np_test_accs,axis=0)

          prev_data_training_cost = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/b4h_training_cost.txt',ndmin=2)
          np_training_costs = np.append(prev_data_training_cost,np_training_costs,axis=0)

          prev_data_training_accs = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/b4h_training_acc.txt',ndmin=2)
          np_training_accs = np.append(prev_data_training_accs,np_training_accs,axis=0)

          prev_times = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/b4h_times.txt',ndmin=2)
          np_times = np.append(prev_times, np_times)

          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b4h_training_cost.txt',np_training_costs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b4h_test_accs.txt',np_test_accs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b4h_training_acc.txt',np_training_accs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b4h_times.txt',np_times)
          
      except Exception:
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b4h_training_cost.txt',np_training_costs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b4h_test_accs.txt',np_test_accs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b4h_training_acc.txt',np_training_accs)
          np.savetxt('/content/gdrive/My Drive/Colab Notebooks/b4h_times.txt',np_times)


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
  plt.ylabel('Training Accuracy')
  plt.legend(legends)

  plt.show()
  sess.close()
    
  
if __name__ == '__main__':
  main()
