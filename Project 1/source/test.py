#
# Project 1, starter code part b
#

import math
import tensorflow as tf
import numpy as np
#use matplotlib because of fucked up
#import pylab as plt
import matplotlib.pyplot as plt


cal_housing = np.loadtxt('cal_housing.data', delimiter=',')

print(cal_housing.shape)
print(cal_housing[1:5])
print(cal_housing[1:5,-1])
print(cal_housing[1:5,-1:-3].shape)
print(cal_housing[1:5,-2:-4])
print(cal_housing[1:5,-2:-4].shape)