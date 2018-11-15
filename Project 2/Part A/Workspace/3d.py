import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure()
ax = plt.axes(projection='3d')

zline = (5,8,15,11,6,9)
yline = (0,1,2,0,1,2)
xline = (0,0,0,1,1,2)

ax.plot3D(xline,yline,zline,'gray')

plt.show()


np.savetxt('test',(xline,yline,zline))