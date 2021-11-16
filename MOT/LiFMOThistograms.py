# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:54:53 2021

@author: marijn

Script fits x and y histograms
"""

#%% Imports
from matplotlib import gridspec
from matplotlib import ticker
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import unravel_index
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.optimize import curve_fit

#%% Variables
cropping_range = 30 # pixels
pixel_size = 4.65e-6 # micron
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

#%% Histograms

# Set up x,y variables from 2D imshow plot defined as twice the cropping range
# Contrary to other script here it is in terms of pixels, because it is used to fit the imshow 
# Which is in pixels
row_range_pixels = np.linspace(0, 2 * cropping_range - 1, 2 * cropping_range)
col_range_pixels = np.linspace(0, 2 * cropping_range - 1, 2 * cropping_range)

# Compute histograms with coordinates x,y
hist_rows = RoI_normalized.sum(axis = 0) 
hist_rows_norm = hist_rows / np.max(hist_rows)

hist_cols = RoI_normalized.sum(axis = 1)
hist_cols_norm = hist_cols / np.max(hist_cols)

#%%Fitting
# Fit Data copied from other MOT LiF script

# Fitting function
def lorentzian(xdata, offset, amplitude, middle, width):
    return offset + amplitude * width / ((xdata - middle)**2 + 0.25 * width**2)

# Initial guessses
amplitude_guess = 1
offset_guess = 5
width_guess = 25
middle_guess = cropping_range
lorentzian_guess = [offset_guess, amplitude_guess, middle_guess, width_guess]

# Fit Lorentzian
fit_params_rows_pixels, cov_rows_pixels = curve_fit(lorentzian, row_range_pixels, hist_rows_norm, p0 = lorentzian_guess)
fit_params_cols_pixels, cov_cols_pixels = curve_fit(lorentzian, col_range_pixels, hist_cols_norm, p0 = lorentzian_guess)

#%% Plotting
fig = plt.figure(figsize = (5, 5))

# Define size of histogram plots 
spacing_axes_grid_rows = 4
spacing_axes_grid_columns = 4
axes_grid = gridspec.GridSpec(spacing_axes_grid_rows, spacing_axes_grid_columns,
                             hspace = 0.2, 
                             wspace = 0.2
                             )

ax_img = plt.subplot(axes_grid[1:4, 0:3])

# horizontal histogram
axHistX = plt.subplot(axes_grid[0:1, 0:3])
axHistX.plot(hist_cols_norm)

# vertical histogram
axHistY = plt.subplot(axes_grid[1:4, 3:4])
axHistY.plot(hist_rows_norm,
             range(len(hist_rows_norm)))

img = ax_img.imshow(RoI_normalized,
                 interpolation = 'nearest',
                 origin = 'lower',
                 vmin = 0.)
img.set_cmap('jet')
ax_img.axis('off')

# Fit
lorentzian_fit_rows = lorentzian(row_range_pixels, *fit_params_rows_pixels)
lorentzian_fit_rows_norm = lorentzian_fit_rows / np.max(lorentzian_fit_rows)

lorentzian_fit_cols = lorentzian(col_range_pixels, *fit_params_cols_pixels)
lorentzian_fit_cols_norm = lorentzian_fit_cols / np.max(lorentzian_fit_cols)

axHistX.plot(row_range_pixels, 
             lorentzian_fit_rows_norm,
             color = 'red')
axHistY.plot(lorentzian_fit_cols_norm,
             col_range_pixels,
             color = 'red')

# Remove tick labels
null_fmt = ticker.NullFormatter()
axHistX.xaxis.set_major_formatter(null_fmt)
axHistX.yaxis.set_major_formatter(null_fmt)

axHistY.xaxis.set_major_formatter(null_fmt)
axHistY.yaxis.set_major_formatter(null_fmt)

# Grid
axHistX.grid()
axHistY.grid()

# Scale
scalebar_object_size = 200e-6 #micron
scalebar_pixels = int(scalebar_object_size / (pixel_size / magnification)) # integer number pixels

scale_bar = AnchoredSizeBar(ax_img.transData,
                           scalebar_pixels, # pixels
                           r'200 $\mu$m', # real life distance of scale bar
                           'lower left', 
                           pad = 1,
                           color = 'white',
                           frameon = False,
                           size_vertical = 1.5)
ax_img.add_artist(scale_bar)

#%% saving
plt.savefig('exports/LiF_MOT_histograms.pdf',
            dpi = 300,
            bbox_inches= 'tight')
plt.show()
