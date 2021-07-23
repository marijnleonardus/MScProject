# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:36:23 2021

@author: Marijn Venderbosch

Script will detect blobs using LoG algorithm. Will show detected blobs by 
overlaying on image plot
"""

import numpy as np
from skimage.feature import blob_log
import matplotlib.pyplot as plt

# Load image from script 'loadmatSaveCamFrame.py'. 
# Image is cropped to region of interest
# Transpose because camera is rotated
image_transposed = np.load('cam_frame_array_cropped.npy')
image = np.transpose(image_transposed)

# Use LoG blog detection. 
# Max_sigma is the max. standard deviation of the Gaussian kernel used. 
# Num_sigma the number of intermediate steps in sigma.
# Threshold determines how easily blobs are detected. 
threshold = 0.1
spots_LoG = blob_log(image, max_sigma = 30, num_sigma = 10, threshold = threshold)

# Check if expected amount of spots is detected
number_spots_expected = 9
number_spots_found = spots_LoG.shape[0]
if number_spots_expected != number_spots_found:
    print('Error: spot finding did not find the expected number of spots')

# Compute radii in the 3rd column by multipling with sqrt(2)
spots_LoG[:, 2] = spots_LoG[:, 2] * np.sqrt(2)

# Find maxima locations and sizes. x and y are swapped becaues tranposed
maxima_y_coordinates = spots_LoG[:, 0]
maxima_x_coordinates = spots_LoG[:, 1]
# Increase sizes crosses for better visibility
factor = 5
sizes = spots_LoG[:, 2] * factor

# Initialize plot
fig, axes = plt.subplots(1, 1, figsize=(5, 4))

# Plot original image and overlay with crosses on spots where blobs are detected
# Radii or cicles are from the gaussian kernels that detected them
axes.set_title('Laplacian of Gaussian Spots')
axes.set_xlabel('Pixels')
axes.set_ylabel('Pixels')
axes.imshow(image)
axes.scatter(maxima_x_coordinates , maxima_y_coordinates, marker = 'x', s = sizes, color = 'r', linewidth = 1)

# Saving and showing
plt.savefig('SpotsFoundUsingLoG.png', dpi = 500, tight_layout = True)
plt.show()