#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:41:31 2021
@author: marijn
Script makes a plot of the laser induced fluorescence from the MOT with
a color overlay and a scalebar
"""

#%% Imports
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import unravel_index
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm


#%% Variables
cropping_range = 80
pixel_size = 4.65e-6
magnification = 0.5      

#%%importing data
# bmp file
image = Image.open('images/mot.bmp')
array = np.array(image) 

# Finding center MOT
max_loc = array.argmax()
indices= unravel_index(array.argmax(), array.shape)

# Cropping                                                     
RoI = array[indices[0] - cropping_range : indices[0] + cropping_range, 
            indices[1] - cropping_range : indices[1] + cropping_range]

# Normalize
RoI_normalized = RoI / np.max(RoI)
             
#%% Plotting
fig, ax = plt.subplots(figsize = (1.8, 1.8))
img = ax.imshow(RoI_normalized)
img.set_cmap('magma')
ax.axis('off')

# Scale
scalebar_object_size = 250e-6 #micron
scalebar_pixels = int(scalebar_object_size / (pixel_size / magnification)) # integer number pixels

scale_bar = AnchoredSizeBar(ax.transData,
                           scalebar_pixels, # pixels
                           r'250 $\mu$m', # real life distance of scale bar
                           'lower left', 
                           pad = 0,
                           color = 'white',
                           frameon = False,
                           size_vertical = 2.5)
ax.add_artist(scale_bar)

# Colorbar
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",
                          size= " 5%",
                          pad = 0.05)

cbar = plt.colorbar(img,
                    ticks = np.linspace(0, 1, 3),
                    extendrect = 'true',
                    cax = cax
                    )

#%%saving
plt.savefig('exports/LiF_MOT.pdf',
            dpi = 300,
            bbox_inches= 'tight')

# Colorbar
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",
                          size= " 5%",
                          pad = 0.05)

cbar = plt.colorbar(img,
                    ticks = np.linspace(0, 1, 3),
                    extendrect='true',
                    cax=cax,
                    )

#%%saving
plt.savefig('LiF_MOT.pdf',
            dpi = 300,
            bbox_inches= 'tight')
