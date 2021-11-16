#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:41:31 2021

@author: marijn

Script makes a plot of the laser induced fluorescence from the MOT with
a color overlay and a scalebar
"""

#%% Imports
from matplotlib import gridspec
from matplotlib import ticker
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import unravel_index
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% Variables
cropping_range = 70 # pixels
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

#%% Histograms
# Set up x,y variables from 2D imshow
xrange = np.linspace(0, cropping_range - 1, cropping_range)
yrange = xrange
xpixels, ypixels = np.meshgrid(xrange, yrange)

# Compute histograms with coordinates x,y
histogram_rows = RoI_normalized.sum(axis = 0)
histogram_columns = RoI_normalized.sum(axis = 1)

             
#%% Plotting
fig = plt.figure(figsize = (5, 5))

# Define size of histogram plots 
spacing_axesGrid_rows = 400
spacing_axesGrid_columns = 80
axesGrid = gridspec.GridSpec(spacing_axesGrid_rows, spacing_axesGrid_columns,
                             hspace = 0.2, 
                             wspace = 0.5
                             )

axImage = plt.subplot(axesGrid[0:300, 20:80])
# horizontal histogram
axHistX = plt.subplot(axesGrid[302:400, 20:76])
axHistX.plot(histogram_columns)
# vertical histogram
axHistY = plt.subplot(axesGrid[6:293, 0:18])
axHistY.plot(histogram_rows)

img = axImage.imshow(RoI_normalized,
                 interpolation = 'nearest',
                 origin = 'lower',
                 vmin = 0.)
img.set_cmap('jet')
axImage.axis('off')

# Remove tick labels
nullfmt = ticker.NullFormatter()
axHistX.xaxis.set_major_formatter(nullfmt)
axHistX.yaxis.set_major_formatter(nullfmt)
axHistY.xaxis.set_major_formatter(nullfmt)
axHistY.yaxis.set_major_formatter(nullfmt)

# Grid
axHistX.grid()
axHistY.grid()

# Scale
scalebar_object_size = 200e-6 #micron
scalebar_pixels = int(scalebar_object_size / (pixel_size / magnification)) # integer number pixels

scale_bar = AnchoredSizeBar(axImage.transData,
                           scalebar_pixels, # pixels
                           r'200 $\mu$m', # real life distance of scale bar
                           'lower left', 
                           pad = 0,
                           color = 'white',
                           frameon = False,
                           size_vertical = 2.5)
axImage.add_artist(scale_bar)

# Colorbar
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(axImage)
cax = divider.append_axes("right",
                          size= " 5%",
                          pad = 0.05)

cbar = plt.colorbar(img,
                    ticks = np.linspace(0, 1, 3),
                    extendrect='true',
                    cax=cax,
                    )


#%% saving
plt.savefig('exports/LiF_MOT_november2.pdf',
            dpi = 300,
            bbox_inches= 'tight')
plt.savefig('exports/LiF.png',
            dpi=300,
            bbox_inches = 'tight')
plt.show()
