import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = np.loadtxt('results.txt')

fig = plt.figure(1)
ax = plt.axes(projection='3d')

ax.set_xlabel('grid search: C1')
ax.set_ylabel('grid search: C2')
ax.set_zlabel('Test Accuracy')

xdata = data[:,0]
ydata = data[:,1]
zdata = data[:,2]

ax.scatter3D(xdata,ydata,zdata,'gray')

fig = plt.figure(2)
ax = plt.axes(projection='3d')

xdata = data[:,0]
ydata = data[:,1]
zdata = data[:,3]

ax.set_xlabel('grid search: C1')
ax.set_ylabel('grid search: C2')
ax.set_zlabel('Duration')

ax.scatter3D(xdata,ydata,zdata,'gray')

plt.show()

