from fnn_part_b_4_L3 import evaluate_fnn_param as L3
from fnn_part_b_4_L3N import evaluate_fnn_param as L3N
from fnn_part_b_4_L4 import evaluate_fnn_param as L4
from fnn_part_b_4_L4N import evaluate_fnn_param as L4N
from fnn_part_b_4_L5 import evaluate_fnn_param as L5
from fnn_part_b_4_L5N import evaluate_fnn_param as L5N
import matplotlib.pyplot as plt
import multiprocessing as mp

epochs = 1000

def worker_thread(param):

    layer, dropout = param

    if dropout:
        result = eval("L{}(epochs,20)".format(layer))
    else:
        result = eval("L{}N(epochs,20)".format(layer))

    return result

def main():

    no_threads = mp.cpu_count()

    params = [(3, True), (3, False), (4, True), (4, False), (5, True),(5,False)]
    p = mp.Pool(processes=no_threads)
    results = p.map(worker_thread, params)

    test_accs = []
    training_accuracy = []
    epoch_time = []
    batch_time = []
    legend = ["{} Layer Network".format(p[0]) if p[1] else "{} Layer Network, No Dropouts".format(p[0]) for p in params]

    for i, result in enumerate(results):
        test_accs.append(result[0])
        training_accuracy.append(result[1])
        epoch_time = [result[2]]
        batch_time = [result[3]]

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

    plt.figure(3)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Epoch Time')
    plt.legend(params)

    plt.figure(4)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Batch Time')
    plt.legend(params)

    plt.show()

if __name__ == "__main__":
    main()

