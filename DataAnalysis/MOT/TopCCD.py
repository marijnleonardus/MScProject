# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:21:01 2021

@author: marijn

loads bmp file from camera.
Plots overlap tweezer with MOT
Tweezer @ 780 because otherwise bandpass filter blocks it

"""

#%% imports

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log

#%% variables

# pixels
magnification = 1
pixel_size = 4.65 #micron

# loading image
location = 'images/overlapTweezer/' 
filename  = 'top camera.bmp'
 
# laplacian of gaussian
nr_max_expected = 1
threshold = 0.1

# RoI
cropping_pixels = 20

#%% load image file

def bmp_import(location, filename):
    bmp_file = Image.open(location + filename)
    array = np.array(bmp_file)
    #array = array[:,:,0]
    return array
    
image = bmp_import(location, filename)

#%% detect maximum and crop to RoI

def RoI_crop(array, cropping_range):
    max_location = blob_log(array,
                            min_sigma = 5,
                            max_sigma = 15,
                            num_sigma = 5,
                            threshold = threshold
                            )
    
    # Check if expected amount of spots is detected
    number_spots_found = max_location.shape[0]
    if nr_max_expected != number_spots_found:
        print('Error: spot finding did not find the expected number of spots')
    
    # RoI size
    max_location_row = int(max_location[:, 1])
    max_location_col = int(max_location[:, 0])
    
    # Crop image to RoI
    RoI = array[
        max_location_col - cropping_range : max_location_col + cropping_range,
        max_location_row - cropping_range : max_location_row + cropping_range
                ]
    return RoI

RoI = RoI_crop(image, cropping_pixels)

#%% plotting and saving

def plot_image(array):
    fig, ax = plt.subplots(figsize = (3,3))
   
    extend = [0 , 2 * pixel_size * magnification * cropping_pixels,
              0 , 2 * pixel_size * magnification * cropping_pixels]
   
    ax.imshow(array,
              cmap = 'viridis',
              origin = 'lower',
              extent = extend
              )
    
    ax.set_xlabel(r'$x$ [$\mu$m]')
    ax.set_ylabel(r'$y$ [$\mu$m]')
    
    markings = 4
    plt.locator_params(axis = "x", nbins = markings)
    plt.locator_params(axis = "y", nbins = markings)
    
plot_image(RoI)
plt.savefig('exports/TopCCDimage.png',
            bbox_inches = 'tight',
            dpi = 300
            )
