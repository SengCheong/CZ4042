import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

epochs = 1000

# adam_test_accs = np.loadtxt('adam_test_accs.txt')
# adam_training_cost = np.loadtxt('adam_training_cost.txt')

# momentum_test_accs = np.loadtxt('momentum_test_accs.txt')
# momentum_training_cost = np.loadtxt('momentum_training_cost.txt')

# rmsprop_test_accs = np.loadtxt('rmsprop_test_accs.txt')
# rmsprop_training_cost = np.loadtxt('rmsprop_training_cost.txt')

standard_test_accs = np.loadtxt('standard_test_accs.txt')
standard_training_cost = np.loadtxt('standard_training_cost.txt')

dropouts_training_cost = np.loadtxt('dropouts_training_cost.txt',ndmin=2)
dropouts_test_accs = np.loadtxt('dropouts_test_accs.txt',ndmin=2)

#legends = ["standard","momentum","rmsprop","adam"]
legends = ["standard"]

for p in range(1,10):
	legends.append("{} keep-prob".format(p/10))

final_test_acc_pts = [standard_test_accs]

for dropout in dropouts_test_accs:
	final_test_acc_pts.append(dropout)

final_training_cost_pts = [standard_training_cost]

for dropout in dropouts_training_cost:
	final_training_cost_pts.append(dropout)


plt.figure(1)
for acc in final_test_acc_pts:
    plt.plot(range(epochs), acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Test Accuracy')
plt.legend(legends)

plt.figure(2)
for acc in final_training_cost_pts:
    plt.plot(range(epochs), acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Cost')
plt.legend(legends)

plt.show()

