# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:41:07 2021

@author: Marijn Venderbosch

This script will compute the azimuthal average around a spot.
Preferably a single spot so the rings are clearly visible. 
"""

# Libraries used
import numpy as np
import scipy.io
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from scipy import optimize 
from scipy.stats import norm
from mpl_toolkits import mplot3d
from matplotlib import cm

""""This windows contains the variables that need to be edited, rest of script
does not have to be edited"""
# Number of spots that we make, to check if spot detection worked
number_spots_expected = 1
# Location of .mat data file
mat_file_location = 'files/3x3 spaced 9 5.mat'
# Threshold on how sensitive spot detection is
threshold = 0.25
# How many pixels do we crop around the spot maxima locations
cropping_range = 10
# magnification from newport objective. This is uncalibrated. 
magnification = 60

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
    np.save('files/cam_frame_cropped.npy', cam_frame_cropped)
 
# execute function. Insert in brackets the .mat filename
load_and_save('files/strehl.mat')
 
image_transposed = np.load('files/cam_frame_cropped.npy')
image = np.transpose(image_transposed)

fig, ax = plt.subplots(1,1)
ax.imshow(image_transposed)


"""The following part will detect maxima using the Laplacian of Gaussian algorithm"""

# Load image from script 'loadmatSaveCamFrame.py'. 
# Image is cropped to region of interest
# Transpose because camera is rotated
image_transposed = np.load('files/cam_frame_cropped.npy')
image = np.transpose(image_transposed)

# Use LoG blog detection. 
# Max_sigma is the max. standard deviation of the Gaussian kernel used. 
# Num_sigma the number of intermediate steps in sigma.
# Threshold determines how easily blobs are detected. 

spots_LoG = blob_log(image, max_sigma = 30, num_sigma = 10, threshold = threshold)
# Save result to be used by other script
np.save('files/spots_LoG', spots_LoG)

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
fig, axes = plt.subplots(1, 1, figsize=(5, 4))

# Plot original image and overlay with crosses on spots where blobs are detected
# Radii or cicles are from the gaussian kernels that detected them
axes.set_title('Laplacian of Gaussian Spots')
axes.set_xlabel('Pixels')
axes.set_ylabel('Pixels')
axes.imshow(image)
axes.scatter(maxima_x_coordinates , maxima_y_coordinates, marker = 'x', s = sizes, color = 'r', linewidth = 1)

# Saving and showing
plt.savefig('exports/SpotsFoundUsingLoG.png', dpi = 500, tight_layout = True)


""""radial profile
We want to compute the azimuthal average"""

# Image dimensions are 'a x b' 
a = image.shape[0]
b = image.shape[1]

# Make new coordinate system centered around the maximum and compute radii from center
[XX, YX] = np.meshgrid(np.arange(b) - maxima_x_coordinates, np.arange(a) - maxima_y_coordinates)
R = np.sqrt(XX**2 + YX**2)

# We need an array to keep track of how many pixels we are from the center
rad = np.arange(1, np.max(R), 1)

# We initialize an empty matrix that will store intensity on each ring
intensity = np.zeros(len(rad))

# How thick the rings are (the mask). Increase for more averaging and data points. 
bin_size = 2

# Define rings around the center with thickness binsize 
# For each ring/mask, select only those pixels that are within the ring. 
# And compute average intensity on the ring
index = 0
for m in rad:
    ring = (np.greater(R, m - bin_size) & np.less(R, m + bin_size))
    pixels_inside = image[ring]
    intensity[index] = np.mean(pixels_inside)
    index +=1
    
# Create figure and add subplot
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111)

# Plot data
ax.grid()
ax.scatter(rad, intensity, linewidth = 1)

# Edit axis labels
ax.set_xlabel('Radial Distance')
ax.set_ylabel('Average Intensity')

# Turns out we don't need the edges
ax.set_xlim(0, 30)

# Save figure
plt.savefig('exports/Azimuthal average strehl ratio,pdf', dpi = 500, tight_layout = False)

