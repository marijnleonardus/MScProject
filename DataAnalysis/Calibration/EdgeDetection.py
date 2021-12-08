# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:25:16 2021

@author: Marijn Venderbosch

Script for 1D edge detection to calibrate camera using distance between lines
in resolution target. 
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from skimage import feature, filters
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

#%% Loading data


""""This windows contains the variables that need to be edited, rest of script
does not have to be edited"""

# Location of .mat data file
mat_file_location = 'files/day2.mat'
# magnification from newport objective. This is uncalibrated. 


"""" The following function will take the .mat file and export a grayscale numpy array
with similar dimensions as the accompanying screenshot"""
def load_and_save(mat_file):
    mat_file = scipy.io.loadmat(mat_file)
    
    # the cam_frame entry contains the raw camera data
    cam_frame = mat_file['cam_frame']
    
    # Crop the camera window to the region of interest, such that the dimensions
    # match the screenshot that is saved as well. The coordinates used to crop 
    # the screenshot are stored in the .mat directory
    # coordinates need to convert from a 1x1 array to an integer
    cam_x_min = int(mat_file['cam_x_min'])
    cam_x_max = int(mat_file['cam_x_max'])

    cam_y_min = int(mat_file['cam_y_min'])
    cam_y_max = int(mat_file['cam_y_max'])
    
    # Cropping the array using the provided coordinates
    cam_frame_cropped = cam_frame[cam_x_min : cam_x_max , cam_y_min : cam_y_max]
    
    return cam_frame_cropped
 
# execute function. Insert in brackets the .mat filename
image_snip = load_and_save(mat_file_location)

"""Show the image around the region of interest"""
# We only use the first 400 pixels becaues the camera is burnt around the middle
image = image_snip[500:1000, :]


#%% Plotting

# Initialize plot
fig, ax = plt.subplots( figsize = (6, 3))


# Major and minor ticks for x,y as well as labels
#ax.xaxis.set_major_locator(MultipleLocator(100))
#ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.set_xlabel('pixels')

#ax.yaxis.set_major_locator(MultipleLocator(100))
#ax.yaxis.set_minor_locator(MultipleLocator(25))
ax.set_ylabel('pixels')

# Grid
ax.grid(color = 'white')

# Plot
ax.imshow(image,
          cmap = 'Greys_r')

fig.savefig('exports/LineSpacingCalibration.pdf', 
            dpi = 200, 
            pad_inches = 0,
            bbox_inches = 'tight')

#%% Edge detection

# - We smooth with a Gaussian
# - We compute the derivative
# - Select only maxima of the derivative above a certain threshold

# For better signal/noise ratio and because we only care about 1D, compute
# histogram over all rows
row = image[420, :]


# Variables
sigma = 3
threshold = 2.5

# Gaussian blur, edges cover about 9 pixels. Compute derivative
blurred = gaussian_filter1d(row, 
                            sigma = 3)

derivative = np.gradient(blurred)

# Find peaks above certain height
edges, peak_heights = find_peaks(derivative,
                                 height = threshold)

print("the edge locations are: " + str(edges))

