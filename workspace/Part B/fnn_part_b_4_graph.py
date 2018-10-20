from fnn_part_b_4_L3 import evaluate_fnn_param as L3
from fnn_part_b_4_L3N import evaluate_fnn_param as L3N
from fnn_part_b_4_L4 import evaluate_fnn_param as L4
from fnn_part_b_4_L4N import evaluate_fnn_param as L4N
from fnn_part_b_4_L5 import evaluate_fnn_param as L5
from fnn_part_b_4_L5N import evaluate_fnn_param as L5N
import matplotlib.pyplot as plt
import multiprocessing as mp


def worker_thread(param):

    layer, dropout = param

    if dropout:
        result = eval("L{}()".format(layer))
    else:
        result = eval("L{}N()".format(layer))

    return result

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


