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
from skimage.feature import blob_log
import matplotlib.pyplot as plt

#%% variables

threshold = 0.2
number_spots_expected = 1
cropping_range = 25
pixel_size = 6.5# micron
magnification = 55/4 # f2/f1

#%% Load, crop RoI

location = 'images/MOTonly/' 
filename  = 'andor.bmp'

def bmp_import(location, filename):
    bmp_file = Image.open(location + filename)
    array = np.array(bmp_file)
    return array
    
image = bmp_import(location, filename)


# Find max location
def spot_detection(image):
    
    # Use LoG blog detection. 
    # Max_sigma is the max. standard deviation of the Gaussian kernel used. 
    # Num_sigma the number of intermediate steps in sigma.
    # Threshold determines how easily blobs are detected. 
    
    spots_LoG = blob_log(image,
                         max_sigma = 7,
                         num_sigma = 3, 
                         threshold = threshold
                         )

    # Check if expected amount of spots is detected

    number_spots_found = spots_LoG.shape[0]
    if number_spots_expected != number_spots_found:
        print('Error: spot finding did not find the expected number of spots')
    maxRow = int(spots_LoG[:, 0])
    maxCol = int(spots_LoG[:, 1])
    return maxRow, maxCol

maxRow, maxCol = spot_detection(image)

# crop RoI
def crop_RoI(image, maxRow, maxCol, Range):
    array = image
    array_crop = array[maxRow - Range : maxRow + Range,
                       maxCol - Range : maxCol + Range]
    return array_crop

RoI = crop_RoI(image, maxRow, maxCol, cropping_range)

#%% plotting and saving

fig, ax = plt.subplots()
ax.imshow(RoI,
          #extent = pixel_size / magnification * [-5, 5, -5, 5],
          cmap = 'jet')

