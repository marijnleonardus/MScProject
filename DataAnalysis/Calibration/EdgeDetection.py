# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:25:16 2021

@author: Marijn Venderbosch

Script for 1D edge detection to calibrate camera using distance between lines
in resolution target. 
"""

# Libraries used
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from skimage import feature, filters
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


""""This windows contains the variables that need to be edited, rest of script
does not have to be edited"""

# Location of .mat data file
mat_file_location = '-200.mat'
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
    
    # Save cropped frame as numpy array
    np.save('cam_frame_array_cropped', cam_frame_cropped)
 
# execute function. Insert in brackets the .mat filename
load_and_save(mat_file_location)

# Load image from script 'loadmatSaveCamFrame.py'. 
image_full = np.load('cam_frame_array_cropped.npy')

"""Show the image around the region of interest"""
# We only use the first 400 pixels becaues the camera is burnt around the middle
image = image_full[0:400, :]

# Initialize plot
fig, ax = plt.subplots( figsize = (6,3))
fig.suptitle('45 lines per mm')

# Major and minor ticks for x,y as well as labels
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.set_xlabel('pixels')

ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(25))
ax.set_ylabel('pixels')

# Grid
ax.grid(color = 'white')

# Plot
ax.imshow(image)
fig.savefig('linespacing.pdf', dpi = 300, bbox_inches = 'tight')
plt.show()
plt.tight_layout()

"""
edge detection
The algorithm is straightforward. 
    We smooth with a Gaussian
    We compute the derivative
    Select only maxima of the derivative above a certain threshold
"""

# For better signal/noise ratio and because we only care about 1D, compute
# histogram over all rows
histogram = image.sum(axis = 0)

# Subtract a background
histogram = histogram - np.min(histogram)

# Gaussian blur, edges cover about 9 pixels. Compute derivative
blurred = gaussian_filter1d(histogram, sigma = 9)
derivative = np.gradient(blurred)

# Find peaks above certain height
edges, peak_heights = find_peaks(derivative, height = 150)

# Print result
print("the edge locations are: ")
print(edges)

