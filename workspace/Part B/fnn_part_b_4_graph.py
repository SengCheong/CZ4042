from . import fnn_part_b_4_L3 as L3
from . import fnn_part_b_4_L3N as L3N
from . import fnn_part_b_4_L4 as L4
from . import fnn_part_b_4_L4N as L4N
from . import fnn_part_b_4_L5 as L5
from . import fnn_part_b_4_L5N as L5N
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def worker_thread(layer,dropout):

    if dropout:
        result = eval("L{}()".format())
    else:
        result = eval("L{}N()".format())


epochs = 1000

no_threads = mp.cpu_count()

params = [(3,True),(3,False),(4,True),(4,False),(5,True)]
p = mp.Pool(processes=no_threads)
results = p.map(worker_thread(), params)

test_accs = []
training_accuracy = []
legend = ["{} Layer Network".format(p[0]) if p[1] else "{} Layer Network, No Dropouts".format(p[0]) for p in params]

for i,result in enumerate(results):
    print("Result: {} - {}".format(i,result[2]))
    test_accs.append(result[0])
    training_accuracy.append(result[1])

plt.figure(1)
for acc in test_accs:
    plt.plot(range(epochs), acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Accuracy against Test Data')
plt.legend(params)

plt.figure(2)
for acc in test_accs:
    plt.plot(range(epochs), acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Accuracy against Test Data')
plt.legend(params)

plt.show()


