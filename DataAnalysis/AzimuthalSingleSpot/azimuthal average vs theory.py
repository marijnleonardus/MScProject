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
from scipy.special import jv

""""This windows contains the variables that need to be edited, rest of script
does not have to be edited"""

# magnification from newport objective. This is uncalibrated. 
magnification = 71.1
# Threshold for LoG detection
threshold = 0.2
# Number of spots expected for LoG. Since we don't use a pattern of spots set to 1
number_spots_expected = 1
# mat file

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
    cam_frame_cropped = np.transpose(cam_frame[cam_x_min : cam_x_max , cam_y_min : cam_y_max])
    return cam_frame_cropped
 
# execute function. Insert in brackets the .mat filename
image0 = load_and_save('files/0.mat')
image01 = load_and_save('files/minus01.mat')
image025 = load_and_save('files/minus025.mat')
 
"""The following part will detect maxima using the Laplacian of Gaussian algorithm"""
# Load image from script 'loadmatSaveCamFrame.py'. 
# Image is cropped to region of interest
# Transpose because camera is rotated

# Use LoG blog detection. 
# Max_sigma is the max. standard deviation of the Gaussian kernel used. 
# Num_sigma the number of intermediate steps in sigma.
# Threshold determines how easily blobs are detected. 
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
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))

    # Plot original image and overlay with crosses on spots where blobs are detected
    # Radii or cicles are from the gaussian kernels that detected them
    axes.set_title('Laplacian of Gaussian Spots')
    axes.set_xlabel('Pixels')
    axes.set_ylabel('Pixels')
    axes.imshow(image)
    axes.scatter(maxima_x_coordinates , maxima_y_coordinates, marker = 'x', s = sizes, color = 'r', linewidth = 1)
    
    # return for next script
    maximum_locs = np.array([maxima_x_coordinates, maxima_y_coordinates])
    return maximum_locs

# Execute function and save maxima parameters
maxima0 = spot_detection(image0)
maxima01 = spot_detection(image01)
maxima025 = spot_detection(image025)

""""radial profile
We want to compute the azimuthal average"""

# We need an array to keep track of how many pixels we are from the center
rad = np.arange(1, 101, 1)
# convert from pixels to um using magnification
rad_m = rad *10**(-6) * 4.65 / magnification

def azimuthal_average(image, maxima):
    # Image dimensions are 'a x b' 
    a = image.shape[0]
    b = image.shape[1]
    
    # Make new coordinate system centered around the maximum and compute radii from center
    [XX, YX] = np.meshgrid(np.arange(b) - maxima[0], np.arange(a) - maxima[1])
    R = np.sqrt(XX**2 + YX**2)
    
    # We initialize an empty matrix that will store intensity on each ring
    intensity = np.zeros(len(rad))
    
    # How thick the rings are (the mask). Increase for more averaging and data points. 
    bin_size = 1
    
    # Define rings around the center with thickness binsize 
    # For each ring/mask, select only those pixels that are within the ring. 
    # And compute average intensity on the ring
    index = 0
    for m in rad:
        ring = (np.greater(R, m - bin_size) & np.less(R, m + bin_size))
        pixels_inside = image[ring]
        intensity[index] = np.mean(pixels_inside)
        index +=1
    
    intensity_normalized = intensity / np.max(intensity)
    return intensity_normalized

measurement0 = azimuthal_average(image0, maxima0)
measurement01 = azimuthal_average(image01, maxima01)
measurement025 = azimuthal_average(image025, maxima025)

"""from tweezer vs airy script"""
w_i = 0.002
R = 0.002
lam = 780* 10**(-9)
k = 2*np.pi / lam
f = 0.004

# Radius in focal plane, as well as initializing empty tweezer matrix
tweezer_matrix = []

# Tweezer integral: r' is r_prime
def tweezer(r_prime):
    integral = np.exp(-r_prime**2 / w_i**2) * jv(0, k * r_prime * rad_m[i] / f) * r_prime
    return integral

# Compute numerical integral as a function of r', save result by appending list
for i in range(len(rad_m)):
    result , error = scipy.integrate.fixed_quad(tweezer, 0, w_i)
    tweezer_matrix.append(result)

# Convert to np array, compute intensity, normalize
tweezer_array = np.array(tweezer_matrix)    
tweezer_intensity = abs(tweezer_array)**2
tweezer_intensity_normalized = tweezer_intensity / np.max(tweezer_intensity)

# Plot tweezer
fig, ax = plt.subplots(1,1, figsize = (6, 4))
ax.grid()

# rescale x axis to show um instead of m
rad_microns = rad_m * 10**6

ax.plot(rad_microns, tweezer_intensity_normalized, label = 'tweezer')
ax.scatter(rad_microns, measurement0,
           label = 'no correction', 
           color = 'red', 
           s = 5,
           marker ='X'
           )
#ax.scatter(rad_microns, measurement01, label = '0.1 spherical correction', color = 'orange')
#ax.scatter(rad_microns, measurement025, label = '0.25 spherical correction', color = 'green')

ax.set_xlabel(r'radial coordinate in focal plane $r$ [$\mu$m]')
ax.set_ylabel('normalized intensity [a.u.]')
ax.legend()
ax.set_xlim(0,2)

