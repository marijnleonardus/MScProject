# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:21:01 2021

@author: marijn

Plots bmp file from Andor sCMOS camera
Export image as BMP using GUI application from them
"""

#%% imports

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% variables

threshold = 0.2
number_spots_expected = 1
cropping_range = 15
pixel_size = 6.5# micron
magnification = 55/4 # f2/f1

# center
maxCol = 62
maxRow = 48

# background
background = 10

dimension = '2x2'

#%% Load, crop RoI

location = 'images/26january/' 
filename  = '10avg.bmp'

def bmp_import(location, filename):
    bmp_file = Image.open(location + filename)
    array = np.array(bmp_file)
    return array
    
image = bmp_import(location, filename)

# crop RoI
def crop_RoI(image, maxRow, maxCol, Range):
    array = image
    array_crop = array[maxRow - Range : maxRow + Range,
                       maxCol - Range : maxCol + Range]
    return array_crop

RoI = crop_RoI(image, maxRow, maxCol, cropping_range)
RoI = RoI / np.max(RoI)



#%% plotting and saving

fig, ax = plt.subplots(figsize = (3,2.5))
plot = ax.imshow(RoI, cmap = 'gnuplot2')
ax.axis('off')

#cb = fig.colorbar(plot, shrink = 0.9, ticks =[0,0.5,1])

# annotate
ax.annotate("10 avg.", xy = (0.6, 0.9), xycoords = "axes fraction", fontweight = 'bold', fontsize = 12, color = 'white')

plt.savefig('exports/'+dimension+'fluoresence10avg.pdf',
            pad_inches = 0,
            bbox_inches = 'tight')

