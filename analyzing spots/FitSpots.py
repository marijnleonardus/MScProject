# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:01:42 2021

@author: Marijn Venderbosch

This script crops the spots around the locatins found by the LoG algorithm. 
Subsequently it fits 2D gaussians around the spot locations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize 

# Import image as well as spot locations as found by LoG
image = np.load('cam_frame_array_cropped.npy')
spot_locations = np.load('spots_LoG.npy')

# Cropping ranges: how many pixels left/right and up/down are we considering from the max. 
cropping_range = 5

# Crop size: the amount of pixels 
crop_size_pixels = 2 * cropping_range + 1

# x and y maxima. Transposed so y is the first column
spots_y_coordinates = spot_locations[:, 0]
spots_x_coordinates = spot_locations[:, 1]

# Amount spots
amount_spots = len(spots_x_coordinates)

# Find the lower and upper limits for x and y using the aforemenionted cropping range
# Force every entry to be an integer
lower_limit_x = list(spots_x_coordinates - cropping_range)
lower_limit_x = [int(item) for item in lower_limit_x]

upper_limit_x = list(spots_x_coordinates + cropping_range + 1)
upper_limit_x = [int(item) for item in upper_limit_x]

lower_limit_y = list(spots_y_coordinates - cropping_range)
lower_limit_y = [int(item) for item in lower_limit_y]

upper_limit_y = list(spots_y_coordinates + cropping_range + 1)
upper_limit_y = [int(item) for item in upper_limit_y]

# Define number of subplots to make. E.g. for 9 subplots we want 3 x 3 to we take the 
# square root of the number of subplots
amount_subplots = int(np.sqrt(amount_spots))

# Initialize plot o
fig, axes = plt.subplots(amount_subplots, amount_subplots, figsize = (5, 6), sharex = True, sharey = True)
fig.suptitle('Spots Cropped Around Maxima (pixels)')
# To be able to sum over axes it needs to be raveled
ax = axes.ravel()

# Normalization
maximum_spot_intensity = np.max(image)

# Make an empty list, each entry being an empty array 
spots_cropped = [0] * amount_spots

# Crop around each individual spot and store in the list
for j in range(amount_spots):
    spots_cropped[j] = image[lower_limit_x[j]:upper_limit_x[j] , lower_limit_y[j]:upper_limit_y[j]]
    # normalize
    spots_cropped[j] = spots_cropped[j] / maximum_spot_intensity
    # Plot each indiviual spot
    ax[j].imshow(spots_cropped[j])
    ax[j].set_title(j+1)
    
# Saving and showing    
plt.savefig('SpotsCropped.png', dpi = 500)
plt.show()
 
# Pixel 1D matrices (discreet)
pixels_x = np.arange(0, crop_size_pixels, 1)
pixels_y = np.arange(0, crop_size_pixels, 1)
# 2D matrix
x, y = np.meshgrid(pixels_x, pixels_y)

def two_D_gaussian(X, amplitude, x0, y0, sigma_x, sigma_y):
    # We define the function to fit: a particular example of a 2D gaussian
    # indepdent variables x,y are passed as a single variable X (curve_fit only 
    # accepts 1D fitting, therefore the result is raveled 
    x, y = X
    x0 = float(x0)
    yo = float(y0)    
    exponent = -1 / (2 * sigma_x)**2 * (x - x0)**2 + -1 / (2 * sigma_y)**2 * (y - y0)**2
    intensity = amplitude * np.exp(exponent)
    return intensity.ravel()

# Initial values. The fitting algorithm needs an initial guess. Esimated from 
# plot of the spot. 
initial_guess = (1, cropping_range, cropping_range, cropping_range / 3, cropping_range / 3)

# In the for loop, every iteration we want to store data
# We need to initialize empty lists to store these variables 
fit_parameters = [0] * amount_spots
covariance = [0] * amount_spots
spot_raveled = [0] * amount_spots
sigma = [0] * amount_spots
trap_depth = [0] * amount_spots
max_Gauss_locations = [0] * amount_spots

# For each picture do a 2D Gaussian fit
for k in range(amount_spots):
    # the images containing the spots need to be raveled 
    # the 2D fit can only iterate over one direction
    spot_raveled[k] = spots_cropped[k].ravel()
    # Perform the 2D fit, using the Gaussian function and initial guess
    params, covar = optimize.curve_fit(two_D_gaussian, (x, y), spot_raveled[k], p0 = initial_guess)
    # Store in single variable containing all data: fit_parameters
    # not to be confused with 'params' which does not iterate: this is only from one fit
    fit_parameters[k] = params
    # Store sigma, trap depth as well as max. locations
    sigma[k] = (params[3] + params[4]) / 2
    trap_depth[k] = params[0]
    max_Gauss_locations[k] = [params[1], params[2]]
    
# Store trap depths as well as sigma's in a numpy array
np.save('sigma', sigma)
np.save('trap_depth', trap_depth)
       
