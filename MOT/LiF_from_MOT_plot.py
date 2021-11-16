#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:41:31 2021

@author: marijn

Script makes a plot of the laser induced fluorescence from the MOT with
a color overlay and a scalebar

Also exports sum over rows and columms and fits a Voigt profile through it
"""

#%% Imports
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import unravel_index
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.optimize import curve_fit


#%% Variables
cropping_range = 60 # pixels
pixel_size = 4.65e-6 #microns
magnification = 0.5      

#%%importing data
# bmp file containing MOT image
image = Image.open('images/gain1exp10_2.bmp')
array = np.array(image) 

# Finding center MOT
max_loc = array.argmax()
indices= unravel_index(array.argmax(), array.shape)

# Cropping                                                     
RoI = array[indices[0] - cropping_range : indices[0] + cropping_range, 
            indices[1] - cropping_range : indices[1] + cropping_range]

# Normalize
RoI_normalized = RoI / np.max(RoI)

# Set up x,y variables from 2D imshow plot defined as twice the cropping range
# Multiply with magnification to get real size instead of pixels
# Multiply by 1000 to plot in mm instead of m
RowRange = np.linspace(-cropping_range, cropping_range - 1, 2*cropping_range) * pixel_size / magnification * 10e3 
ColRange = np.linspace(-cropping_range, cropping_range - 1, 2*cropping_range) * pixel_size / magnification * 10e3

# Compute histograms with coordinates x,y
HistRows = RoI_normalized.sum(axis = 0) 
HistRowsNorm = HistRows / np.max(HistRows)

HistCols = RoI_normalized.sum(axis = 1)
HistColsNorm = HistCols / np.max(HistCols)

#%% Fitting
# Fitting function
def Gaussian(x, offset, amplitude, sigma):
    return offset + amplitude * np.exp(-x**2 / (2 * sigma**2))

# Initial guess
amplitude_guess = 1
offset_guess = 0.1
sigma_guess = 0.2
guess = [offset_guess, amplitude_guess, sigma_guess]

poptRows, pcovRows = curve_fit(Gaussian, RowRange, HistRowsNorm, p0 = guess)
poptCols, pcovCols = curve_fit(Gaussian, ColRange, HistColsNorm, p0 = guess)

#%% Plot histograms over rows and columns
figSum, (axRow, axCol) = plt.subplots(nrows = 1,
                                      ncols = 2,
                                      sharey = True,
                                      figsize = (7,3))
# Grid
axRow.grid()
axCol.grid()

# Sum over rows
axRow.scatter(RowRange,
              HistRowsNorm,
              s = 4)
axRow.set_xlabel('Horizontal plane camera [mm]')
axRow.set_ylabel('Normalized pixel counts [a.u.]')

# Sum over columns
axCol.scatter(ColRange,
              HistColsNorm,
              s = 4)
axCol.set_xlabel('Vertical plane camera [mm]')

# Plot fit
axRow.plot(RowRange,
           Gaussian(RowRange, *poptRows),
           color = 'red')
axCol.plot(ColRange,
           Gaussian(ColRange, *poptCols),
           color = 'red')
             
#%% Plot MOT fluoresence image
fig = plt.figure(figsize = (4, 3))
ax = plt.subplot()

# MOT fluoresence image
img = ax.imshow(RoI_normalized, 
                interpolation = 'nearest',
                origin = 'lower',
                vmin = 0.)
img.set_cmap('jet')
ax.axis('off')

# Colorbar
cb = plt.colorbar(img,
                  ax = ax,
                  ticks = np.linspace(0, 1, 5),
                  orientation = 'vertical')

# Scalebar
scalebar_object_size = 200e-6 #micron
scalebar_pixels = int(scalebar_object_size / (pixel_size / magnification)) # integer number pixels

scale_bar = AnchoredSizeBar(ax.transData,
                           scalebar_pixels, # pixels
                           r'200 $\mu$m', # real life distance of scale bar
                           'lower left', 
                           pad = 0,
                           color = 'white',
                           frameon = False,
                           size_vertical = 2.5)
ax.add_artist(scale_bar)

#%% Saving
plt.savefig('exports/LiF_MOT_november.pdf',
            dpi = 300,
            bbox_inches= 'tight')
plt.show()
