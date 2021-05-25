# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:25:14 2021

@author: Marijn Venderbosch
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 40
X = np.linspace(-2, 2, N)
Y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(X, Y)

# Parameters for 2D Gaussian
mean_x = 0
mean_y = 0

sigma_x = 1
sigma_y = 2

amplitude = 1

# Making the 2D gaussian
def multivariate_gaussian(X, Y, mean_x, mean_y, amplitude, sigma_x, sigma_y):
    return amplitude * np.exp(-1/(2*sigma_x**2) * (X-mean_x)**2) * np.exp(-1/(2*sigma_y**2) * (Y-mean_y)**2)

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(X,Y,mean_x,mean_y,amplitude,sigma_x,sigma_y)

# Make and show plots

# 3D surface plot
fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(1,2,1,projection='3d')
ax1.plot_surface(X, Y, Z)
ax1.set_title("2d")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# 2D contour plot
ax2 = fig.add_subplot(1,2,2)
ax2.contour(X,Y,Z)
ax2.set_title('3d')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.show()
