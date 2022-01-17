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
fig, (ax1,ax2) = plt.subplots(nrows = 1,
                              ncols = 2, 
                              #tight_layout = True,
                              figsize = (9, 2.8)
                              )
                              
maskPlot = ax1.imshow(maskArray, 
                      cmap = 'gray')
ax1.set_xlabel(r'$x$ [pixels]')
ax1.set_ylabel(r'$y$ [pixels]')


twoDplot = ax2.imshow(patternCrop,
                      cmap = 'gray')
ax2.set_xlabel(r'$x$ [focal units]')
ax2.set_ylabel(r'$y$ [focal units]')

# colorbar
#plt.colorbar(twoDplot,
  #           shrink = 0.5,
  #           pad = 0.025,
   #          aspect = 22)

# annotate
ax1.annotate("(a)", xy = (0.5, -0.32), xycoords = "axes fraction", fontweight = 'bold', fontsize = 10)
ax2.annotate("(b)", xy = (0.5, -0.32), xycoords = "axes fraction", fontweight = 'bold', fontsize = 10)

# saving
plt.savefig('exports/MaskAndComputedPattern.pdf',
            dpi = 100,
            pad_inches = 0,
            bbox_inches = 'tight'
            )
