#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 15:49:08 2022

Script plots computed pattern from GSW algorithm as well as phasemask that provides it

@author: marijn
"""

#%% Imports
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% load data

# Load calculated pattern
pattern = Image.open('files/7x7_calc_pattern.bmp')
patternGrey = pattern.convert('L')

patternArray = np.array(patternGrey) / 255

# crop
def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = int(x / 2 - (cropx / 2))
    starty = int(y / 2 - (cropy / 2))  
    return img[starty : starty + cropy, startx : startx + cropx]

patternCrop = crop_center(patternArray, 80, 50)

# load phasemask
mask = Image.open('files/7x7_mask.bmp')
maskArray = np.array(mask)

#%% Ploting
fig, (ax1,ax2) = plt.subplots(1, 2, 
                              tight_layout = True,
<<<<<<< Updated upstream
                              figsize = (12, 3.5))
=======
                              figsize = (7.8, 3.5*2/3))
>>>>>>> Stashed changes

maskPlot = ax1.imshow(maskArray, 
                      cmap = 'gray')
ax1.set_xlabel(r'$x$ [pixels]')
ax1.set_ylabel(r'$y$ [pixels]')

<<<<<<< Updated upstream
ax1.text(-200,
         50,
         r'a)',
         fontsize = 14,
         fontweight = 'bold'
         )

twoDplot = ax2.imshow(patternCrop)
ax2.set_xlabel(r'$x$ [focal units]')
ax2.set_ylabel(r'$y$ [focal units]')

<<<<<<< Updated upstream
ax2.text(-9,
         1.8,
         r'b)',
         fontsize = 14,
         fontweight = 'bold'
         )

plt.savefig('exports/MaskAndComputedPattern.pdf',
            dpi = 100,
            pad_inches = 0,
            bbox_inches = 'tight'
            )
