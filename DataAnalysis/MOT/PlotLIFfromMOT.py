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
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from scipy.odr import ODR, Model, RealData

#%% Variables

cropping_range = 50 # pixels
pixel_size = 4.65e-6 #microns
magnification = 0.5     

# gridspec setup
w = 330
h = 20000

#%%importing data

# bmp file containing MOT image
file_location = 'U:/KAT1/Images/MOT Images/2022-01-12'
file_name = 'MOT_10.bmp'
image = Image.open(file_location + str('/') + file_name)
array = np.array(image) 

# Finding center MOT
max_loc = array.argmax()
indices= unravel_index(array.argmax(), array.shape)

# Cropping                                                     
RoI = array[indices[0] - cropping_range : indices[0] + cropping_range, 
            indices[1] - cropping_range : indices[1] + cropping_range]

# Normalize
RoI = RoI / np.max(RoI)

# Compute histograms with coordinates x,y
HistRows = RoI.sum(axis = 1) 
HistRows = HistRows / np.max(HistRows)

HistCols = RoI.sum(axis = 0)
HistCols = HistCols / np.max(HistCols)

#%% Plot MOT fluoresence image

# Initialize gridspec

fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(4, 4, hspace = 0,
                       wspace = 0, 
                       figure = fig, 
                       height_ratios = [1, h, 1, h / 4.8], width_ratios = [1, w, 1, w / 4.28])

#gs = fig.add_gridspec(2, 4, hspace=0, wspace=0,height_ratios=[3,1],width_ratios=[1,4,1,1])
#(ax1, ax2), (ax3, ax4) = gs.subplots()
ax1 = plt.subplot(gs[0 : 3, 0 : 3])
ax2 = plt.subplot(gs[1, 3])
ax3 = plt.subplot(gs[3, 1])
ax4 = plt.subplot(gs[3, 3])

img = ax1.imshow(RoI, 
                interpolation = 'nearest',
                origin = 'lower',
                vmin = 0.)
img.set_cmap('magma')
ax1.axis('off')

# Scalebar
scalebar_object_size = 100e-6 #micron
scalebar_pixels = int(scalebar_object_size / (pixel_size / magnification)) # integer number pixels

scale_bar = AnchoredSizeBar(ax1.transData,
                           scalebar_pixels, # pixels
                           r'100 $\mu$m', # real life distance of scale bar
                           'lower left', 
                           pad = 0,
                           color = 'white',
                           frameon = False,
                           size_vertical = 2.5)
ax1.add_artist(scale_bar)

#%% Fitting

# Fitting function
def Gaussian(x, offset, amplitude, middle, width):
    return offset + amplitude * np.exp(-0.5 * ((x - middle)/ width)**2)

# Gaussian initial guess fit
offset_guess = 0.2
amplitude_guess = 0.8
middle_guess = 5
width_guess = 15
guess = [offset_guess, amplitude_guess, middle_guess, width_guess]

# independent variales for fitting (pixels)
pixels_x = np.linspace(-cropping_range, cropping_range - 1, 2 * cropping_range)
pixels_y = np.linspace(-cropping_range, cropping_range - 1, 2 * cropping_range)

# Fit Gaussian
poptRows, pcovRows = curve_fit(Gaussian, pixels_y, HistRows, p0 = guess)
poptCols, pcovCols = curve_fit(Gaussian, pixels_x, HistCols, p0 = guess)


#%% Plot sums and fits

# Independent variables for plotting
x_coordinate = np.linspace(-cropping_range, cropping_range - 1, 2 * cropping_range) * pixel_size / magnification * 10e3
y_coordinate = np.linspace(-cropping_range, cropping_range - 1, 2 * cropping_range) * pixel_size / magnification * 10e3

# Sum over rows
ax2.scatter(-np.flip(HistRows), pixels_y * pixel_size / magnification * 10e2,
            s = 3)            

ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.set_xticks([])


ax2.plot(-np.flip(Gaussian(pixels_y, *poptRows)), pixels_y * pixel_size / magnification * 10e2,
         color = 'r',
         linewidth = 1)
ax2.set_ylabel(r'$y$ [mm]')

# Sum over columns
ax3.scatter(pixels_x * pixel_size / magnification * 10e2, HistCols,
            s = 4)
ax3.set_yticks([])


ax3.plot(pixels_x * pixel_size / magnification * 10e2, Gaussian(pixels_x, *poptCols),
         color = 'r',
         linewidth = 1)
ax3.set_xlabel(r'$x$ [mm]')

# Corner bottom right
ax4.axis('off')

#%% Saving

plt.savefig('exports/FluoresenceAndFits.pdf',
            dpi = 300, 
            pad_inches = 0,
            bbox_inches = 'tight')

sigma_x = np.round(poptCols[3] * pixel_size / magnification * 10e5)
print(sigma_x)
sigma_y = np.round(poptRows[3] * pixel_size / magnification * 10e5)
print(sigma_y) 
