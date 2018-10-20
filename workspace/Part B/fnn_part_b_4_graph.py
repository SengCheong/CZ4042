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
    training_accs = []
    epoch_time = []
    batch_time = []
    legend = ["{} Layer Network, Dropouts".format(p[0]) if p[1] else "{} Layer Network, No Dropouts".format(p[0]) for p in params]

    for i, result in enumerate(results):
        training_accs.append(result[0])
        test_accs.append(result[1])
        epoch_time.append(result[2])
        batch_time.append(result[3])

    plt.figure(1)
    for acc in test_accs:
        plt.plot(range(epochs), acc)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test Data Error')
    plt.legend(params)

    plt.figure(2)
    for acc in training_accs:
        plt.plot(range(epochs), acc)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Training Data Error')
    plt.legend(params)

    plt.figure(3)
    for layer, epoch_time in zip([p[0] for p in params],epoch_time):
        plt.plot(layer,epoch_time ,'o')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Layers')
    plt.legend(params)

    plt.figure(4)
    for layer, epoch_time in zip([p[0] for p in params],batch_time):
        plt.plot(layer,epoch_time ,'o')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Batch Time')
    plt.legend(params)

    plt.show()

if __name__ == "__main__":
    main()

