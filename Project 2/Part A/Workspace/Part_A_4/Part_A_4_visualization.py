import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

epochs = 1000

adam_test_accs = np.loadtxt('adam_test_accs.txt')


momentum_test_accs = np.loadtxt('momentum_test_accs.txt')

rmsprop_test_accs = np.loadtxt('rmsprop_test_accs.txt')

standard_test_accs = np.loadtxt('standard_test_accs.txt')

original_test_accs = np.loadtxt('5060_test_accs.txt')
   
legends = ["original:50,60","standard:60,80","momentum:60,80","rmsprop:60,80","adam:60,80"]
for p in range(1,10):
	legends.append("{} keep-prob".format(p/10))

final_test_acc_pts = [original_test_accs, standard_test_accs,momentum_test_accs,rmsprop_test_accs,adam_test_accs]

plt.figure(1)
for acc in final_test_acc_pts:
    plt.plot(range(epochs), acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Test Accuracy')
plt.legend(legends)


plt.show()

