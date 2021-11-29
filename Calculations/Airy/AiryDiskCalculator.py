# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:16:16 2021

@author: Marijn Venderbosch

Plots Airy disk in 1D and 2D
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import jv

#%% Variables, trivial becaues dimensionless 

k = 1
R = 1
f = 1

# 2D plot amount of data points per axis ('pixels') and range
data_points = 100
plot_range = 7

#%% Define Airy Functions

def intensity_1D(u):
    field = jv(0, k * R * u / f)
    intensity = abs(field)**2
    return intensity

def intensity_2D(X, Y):
    u = ((X - x_mid)**2 + (Y - y_mid)**2) ** (0.5)
    field = jv(0, k * R * u / f)
    intensity = abs(field)**2
    return intensity

# independent variable u is dimensionless unit in terms of k f / R

#%% Evaluating functions

# 1D

u = np.linspace(0, 10, 100)
intensity_1D = intensity_1D(u)

# 2D

x = np.linspace(-plot_range, plot_range, 2 * data_points + 1)
y = np.linspace(-plot_range, plot_range, 2 * data_points + 1)

X, Y = np.meshgrid(x, y)

x_mid = np.median(x)
y_mid = np.median(y)

intensity_2D = intensity_2D(X, Y)

#%% Plotting

fig, (ax1, ax2) = plt.subplots(ncols = 2,
                       nrows = 1,
                       figsize = (6, 2))
ax1.grid()
ax1.plot(u, intensity_1D)

im = ax2.imshow(intensity_2D,
                extent = [-plot_range, plot_range,
                          -plot_range, plot_range],
                cmap = 'gray')

# Colorbar

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

#%% Showing, saving

plt.savefig('exports/AiryDisk.pdf', 
            dpi = 300,
            bbox_inches = 'tight')

