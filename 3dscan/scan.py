# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:21:11 2021

@author: Marijn Venderbosch

This script plots a 3d scan of the tweezer. 
Dimensions are set using the scanning step size in z-direction
and pixel size in the r direction
"""

# import module
import pandas as pd
import numpy as np
import scipy.io
from skimage.feature import blob_log
import matplotlib.pyplot as plt

# mat file location
mat_files_location = "./files/9 1 attempt/"

# variable
magnification = 50.3
pixel_size = 4.65
threshold = 0.1
number_spots_expected = 1
# amount of space to see around the maximum location
row_cropping_range = 20

# z scan
z_steps_per_image = 50
z_start = -500
z_stop = 500
step = 0.0203

# functions to call
def spot_detection(image):
    spots_LoG = blob_log(image, max_sigma = 30, num_sigma = 10, threshold = threshold)

    # Check if expected amount of spots is detected
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
    #fig, axes = plt.subplots(1, 1, figsize=(5, 4))

    # Plot original image and overlay with crosses on spots where blobs are detected
    # Radii or cicles are from the gaussian kernels that detected them
    #axes.set_title('Laplacian of Gaussian Spots')
    #axes.set_xlabel('Pixels')
    #axes.set_ylabel('Pixels')
    #axes.imshow(image)
    #axes.scatter(maxima_x_coordinates , maxima_y_coordinates, marker = 'x', s = sizes, color = 'r', linewidth = 1)
    
    # return for next script
    maximum_locs = np.array([maxima_x_coordinates, maxima_y_coordinates])
    return maximum_locs

# assign dataset names
filename_list = list(map(str, 
                         np.arange(z_start, z_stop + z_steps_per_image, z_steps_per_image).tolist()
                         ))

# create empty list
cam_frames = []
rows = []

# append datasets into the list
for i in range(len(filename_list)):
    mat_file = scipy.io.loadmat(mat_files_location + filename_list[i] + ".mat")
    
    # Select full camera frame uncropped
    full_frame = mat_file['cam_frame']
    
    # Crop the camera window to the region of interest, such that the dimensions
    # match the screenshot that is saved as well. The coordinates used to crop 
    # the screenshot are stored in the .mat directory
    # coordinates need to convert from a 1x1 array to an integer
    cam_x_min = int(mat_file['cam_x_min'])
    cam_x_max = int(mat_file['cam_x_max'])
    
    cam_y_min = int(mat_file['cam_y_min'])
    cam_y_max = int(mat_file['cam_y_max'])
    
    # Cropping the array using the provided coordinates
    cam_frame = np.transpose(full_frame[cam_x_min : cam_x_max , cam_y_min : cam_y_max])
    
    # Store cropped frame in a list keeping all frames
    cam_frames.append(cam_frame)
    
    # Find maxima locations
    maximum_loc = spot_detection(cam_frame)
    # Store the row, column where this maximum is
    max_row_index = int(maximum_loc[1])
    max_column_index = int(maximum_loc[0])
    
    # Exclude single row where maximum is
    row = cam_frame[max_row_index, :]
        
    # Select cropping range around maximum
    row_cropped = row[max_column_index - row_cropping_range : 
                      max_column_index + row_cropping_range]
    
    # Store the final result that we want to keep, the intensity around the rows
    # as a function of z direction
    rows.append(row_cropped)

# Convert list of rows to array (2D array: r,z)    
array = np.transpose(np.array(rows))  

# Compute aspect ratio for plot
# Z dire
z_dimension = len(filename_list) * z_steps_per_image * step
r_dimension = (2 * row_cropping_range + 1) * pixel_size / magnification
aspect_ratio = r_dimension / z_dimension

# Initialize plot
fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.set_aspect(0.1)
ax.imshow(array,
          aspect = 1,
          extent = [-z_dimension/2, z_dimension/2,
                    -r_dimension/2,r_dimension/2
              ])
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_xlabel(r'z-direction [$\mu$m]')

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_ylabel(r'Radial direction [$\mu$m]')

# We really don't need the first and latest iamges
ax.set_xlim(-8, 8)

# Saving
plt.savefig('exports/3dscan_tweezer.pdf',
            dpi = 300, 
            tight_layout = True)




