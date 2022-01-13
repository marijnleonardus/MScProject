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
cropping_range = 45 # pixels
pixel_size = 4.65e-6 #microns
magnification = 0.5      

#%%importing data
# bmp file containing MOT image
image = Image.open('images/side/MOT_8.bmp')
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
RowRange = np.linspace(-cropping_range, cropping_range - 1, 2*cropping_range) * pixel_size / magnification * 10e2
ColRange = np.linspace(-cropping_range, cropping_range - 1, 2*cropping_range) * pixel_size / magnification * 10e2

# Compute histograms with coordinates x,y
HistRows = RoI_normalized.sum(axis = 0) 
HistRowsNorm = HistRows / np.max(HistRows)

HistCols = RoI_normalized.sum(axis = 1)
HistColsNorm = HistCols / np.max(HistCols)

#%% Fitting
# Fitting functions
def Lorentzian(x, offset, amplitude, middle, width):
    return offset + amplitude * width / ((x - middle)**2 + .25 * width**2)


# Lorentzian initial guess fit
amplitude_guess = 1
offset_guess = 0.1
width_guess = 1
middle_guess = 0
LorentzianGuess = [offset_guess, amplitude_guess, middle_guess, width_guess]

# Fit Lorentzian
poptRows, pcovRows = curve_fit(Lorentzian, RowRange, HistRowsNorm, p0 = LorentzianGuess)
poptCols, pcovCols = curve_fit(Lorentzian, ColRange, HistColsNorm, p0 = LorentzianGuess)

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
              s = 7)
axRow.set_xlabel(r'$x$ [mm]')
axRow.set_ylabel(r'Counts [a.u.]')

# Sum over columns
axCol.scatter(ColRange,
              HistColsNorm,
              s = 7)
axCol.set_xlabel(r'$y$ [mm]')

# Plot fit
axRow.plot(RowRange,
           Lorentzian(RowRange, *poptRows),
           color = 'red')
axCol.plot(ColRange,
           Lorentzian(ColRange, *poptCols),
           color = 'red')

# Plot fits
plt.savefig('exports/FitSide.pdf',
            dpi = 300,
            pad_inches = 0,
            bbox_inches= 'tight')
             
#%% Plot MOT fluoresence image
fig = plt.figure(figsize = (4, 3))
ax = plt.subplot()

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
plt.savefig('exports/MOTfluoreesnceSide.pdf',
            dpi = 300,
            pad_inches = 0,
            bbox_inches= 'tight')

FWHM_row = abs(poptRows[3])
FWHM_col = poptCols[3]

print('FWHM (rows) is: ' + str(FWHM_row) + ' um')
print('FWHM (columns) is: ' + str(FWHM_col) + ' um')

