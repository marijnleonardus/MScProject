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


#%% Variables
cropping_range = 110
pixel_size = 4.6e-6
magnification = 0.5      

#%%importing data
# bmp file
image = Image.open('mot.bmp')
array = np.array(image) 

# Finding center MOT
max_loc = array.argmax()
indices= unravel_index(array.argmax(), array.shape)

# Cropping                                                     
RoI = array[indices[0] - cropping_range : indices[0] + cropping_range, 
            indices[1] - cropping_range : indices[1] + cropping_range]
             
#%% Plotting
fig, ax = plt.subplots(figsize = (2, 2))
img = ax.imshow(RoI)
img.set_cmap('magma')
ax.axis('off')

# Scale
scalebar_object_size = 100e-6 #micron
scalebar_pixels = int(scalebar_object_size / (pixel_size * magnification)) # integer number pixels

scale_bar = AnchoredSizeBar(ax.transData,
                           scalebar_pixels, # pixels
                           r'100 $\mu$m', # real life distance of scale bar
                           'lower left', 
                           pad = 0,
                           color = 'white',
                           frameon = False,
                           size_vertical = 2.5)
ax.add_artist(scale_bar)

#%%saving
plt.tight_layout()

plt.savefig('LiF_MOT.png',
            dpi = 300,
            bbox_inches= 'tight')
