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
from scipy.special import jn
import matplotlib.ticker as plticker


#%% Variables

# Trivial dimensionless variables

k = 1
R = 1
f = 1

# 2D plot amount of data points per axis ('pixels') and range

data_points = 100
plot_range = 8

#%% Functions

# Jinc function J_1(x) / x 
# It does not calculate when input is zero, then it outputs 'exception' matrix

def jinc(x):
    
    # initialize array returned when x = 0
    # jinc function will be 1/2 here 
    exception = 0.5 * np.ones_like(jn(1,x))
    
    # devide jn(1,x) by x (jinc function)
    return np.divide(jn(1, x), 
                     x, 
                     out = exception, 
                     where = x!=0)

def intensity_1D(u):
    field = jinc(k * R * u / f)
    intensity = abs(field)**2
    return intensity

def intensity_2D(X, Y):
    u = ((X - x_mid)**2 + (Y - y_mid)**2) ** (0.5)
    field = jinc(k * R * u / f)
    intensity = abs(field)**2
    return intensity

#%% Evaluating functions

# 1D
# independent variable u is dimensionless unit in terms of k f / R

# avoid devision by zero
u = np.linspace(0.001, plot_range, 100)
intensity_1D = intensity_1D(u)

# normalize
intensity_1D = intensity_1D / np.max(intensity_1D)

# 2D

def meshgrid_generator(plot_range, data_points):
    
    # Make 2D array with dimensinos as specified under 'variables' 
    x = np.linspace(-plot_range, plot_range, 2 * data_points + 1)
    y = np.linspace(-plot_range, plot_range, 2 * data_points + 1)

    X, Y = np.meshgrid(x, y)
    
    # Return center index array to compute distance to center
    x_mid = int(np.median(x))
    y_mid = int(np.median(y))
    
    return X, Y, x_mid, y_mid

X, Y, x_mid, y_mid = meshgrid_generator(plot_range, data_points)

intensity_2D = intensity_2D(X, Y)
# normalize
intensity_2D = intensity_2D / np.max(intensity_2D)

#%% Plotting

# Initialize plot
fig, (ax1, ax2) = plt.subplots(ncols = 2,
                       nrows = 1,
                       figsize = (6, 2))

# 1D plot
ax1.grid()
ax1.plot(u, intensity_1D)
ax1.set_ylabel('Normalized Intensity')
ax1.set_xlabel(r'$k r^{\prime} R/f$')

# Tick label frequency

loc_x = plticker.MultipleLocator(base = 2) # this locator puts ticks at regular intervals
ax1.xaxis.set_major_locator(loc_x)

loc_y = plticker.MultipleLocator(base = 0.2)
ax1.yaxis.set_major_locator(loc_y)

# 2D plot

im = ax2.imshow(intensity_2D,
                # Labels are in dimensionless units not in pixels
                extent = [-plot_range, plot_range,
                          -plot_range, plot_range],
                cmap = 'jet')
ax2.set_xlabel(r'$k r^{\prime} R/f$')

# Tick label frequency

loc_x_2D = plticker.MultipleLocator(base = 4) # this locator puts ticks at regular intervals
ax2.xaxis.set_major_locator(loc_x_2D)

loc_y_2D = plticker.MultipleLocator(base = 4)
ax2.yaxis.set_major_locator(loc_y_2D)

# Colorbar

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right",
                          size = "8%",
                          pad = 0.1)

plt.colorbar(im, cax=cax)

#%% Showing, saving

plt.savefig('exports/AiryDisk.pdf', 
            dpi = 300,
            bbox_inches = 'tight')
