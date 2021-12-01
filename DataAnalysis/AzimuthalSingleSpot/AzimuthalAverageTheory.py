#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:41:07 2021

@author: Marijn Venderbosch

This script will compute the azimuthal average around a spot.
Preferably a single spot so the rings are clearly visible. 

"""

#%% Imports

import numpy as np
import scipy.io
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from scipy.special import jv

#%% Variables

#magnification from newport objective 
magnification = 71.14 

# Camera pixel size
pixel_microns = 4.65

# Threshold for LoG detection
threshold = 0.2

# Number of spots expected for LoG. Since we don't use a pattern of spots set to 1
number_spots_expected = 1

# mat file
location = 'images/'
file_name = '1x1.mat'

# Parameters for theoretical plot
waist = 2e-3
aperture_radius = 2e-3
wavelength = 820e-9
wavenumber = 2*np.pi / wavelength
focal_length = 4e-3

#%% Functions

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
    cam_frame_cropped = np.transpose(cam_frame[cam_x_min : cam_x_max , 
                                               cam_y_min : cam_y_max])
    return cam_frame_cropped
 
# execute function. Insert in brackets the .mat filename
image_cropped = load_and_save(location + file_name)
 

#%% spot detection

"""The following part will detect maxima using the Laplacian of Gaussian algorithm,
it will plot the spots as well"""

def spot_detection(image):
    
    # Use LoG blog detection. 
    # Max_sigma is the max. standard deviation of the Gaussian kernel used. 
    # Num_sigma the number of intermediate steps in sigma.
    # Threshold determines how easily blobs are detected. 
    
    spots_LoG = blob_log(image,
                         max_sigma = 30,
                         num_sigma = 10, 
                         threshold = threshold
                         )

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
    fig, axes = plt.subplots(1, 1, figsize = (4, 3))

    # Plot original image and overlay with crosses on spots where blobs are detected
    # Radii or cicles are from the gaussian kernels that detected them
    axes.imshow(image)
    
    # Marker
    axes.scatter(maxima_x_coordinates,
                 maxima_y_coordinates, marker = 'x',
                 s = sizes, 
                 color = 'r',
                 linewidth = 1
                 )
    
    # Edits
    axes.set_xlabel('Pixels')
    axes.set_ylabel('Pixels')
    
    # return for next script
    maximum_locs = np.array([maxima_x_coordinates, maxima_y_coordinates])
    return maximum_locs

# Execute function and save maxima parameters
maxima_locations = spot_detection(image_cropped)


#%% azimuthal average
""""radial profile
We want to compute the azimuthal average"""

def radial_distance(image, maxima):
    # We need an array to keep track of how many pixels we are from the center
    radial_distance_pixels = np.arange(1, 101, 1)
    
    # convert from pixels to um using magnification
    radial_distance = radial_distance_pixels * 1e-6 * pixel_microns / magnification
    return radial_distance_pixels, radial_distance

radial_distance_pixels, radial_distance = radial_distance(image_cropped, maxima_locations)

def azimuthal_average(image, maxima):
    # Image dimensions are 'a x b' 
    a = image.shape[0]
    b = image.shape[1]
    
    # Make new coordinate system centered around the maximum and compute radii from center
    [XX, YX] = np.meshgrid(np.arange(b) - maxima[0]
                           , np.arange(a) - maxima[1])
    
    R = np.sqrt(XX**2 + YX**2)
    
    # We initialize an empty matrix that will store intensity on each ring
    intensity_matrix = np.zeros(len(radial_distance))
    
    # How thick the rings are (the mask). Increase for more averaging and data points. 
    bin_size = 1
    
    # Define rings around the center with thickness binsize 
    # For each ring/mask, select only those pixels that are within the ring. 
    # And compute average intensity on the ring
    index = 0
    
    for m in radial_distance_pixels:
        ring = (np.greater(R, m - bin_size) & np.less(R, m + bin_size))
        pixels_inside = image[ring]
        intensity_matrix[index] = np.mean(pixels_inside)
        index +=1
    
    intensity_normalized = intensity_matrix / np.max(intensity_matrix)
    return intensity_normalized

measurement = azimuthal_average(image_cropped, maxima_locations)


#%% numerical integration, theory calculation
"""from tweezer vs airy script"""
# Computes theory result
# Radius in focal plane, as well as initializing empty tweezer matrix
tweezer_matrix = []

# Tweezer integral: r' is r_prime
def tweezer(r_prime):
    prefactor = np.exp(-r_prime**2 / waist**2) 
    integral = prefactor * jv(0, wavenumber * r_prime * radial_distance[i] / focal_length) * r_prime
    return integral

# Compute numerical integral as a function of r', save result by appending list

for i in range(len(radial_distance)):
    result , error = scipy.integrate.fixed_quad(tweezer, 0, waist)
    tweezer_matrix.append(result)

def tweezer_intensity(tweezer_matrix):
    
    # Convert to np array
    tweezer_array = np.array(tweezer_matrix)   
    
    # Calculate intensity
    tweezer_intensity = abs(tweezer_array)**2
    
    # Normalize
    tweezer_intensity_normalized = tweezer_intensity / np.max(tweezer_intensity)
    return tweezer_intensity_normalized

tweezer_intensity = tweezer_intensity(tweezer_matrix)

def airy(f, k, radial_distance, R):
    
    # Definition Airy disk
    airy = f / k / radial_distance * jv(1, k * radial_distance * R / f)
    
    # Calculate intensity
    airy_intensity = abs(airy)**2
    
    # Normalize
    airy_intensity_normalized = airy_intensity / np.max(airy_intensity)
    return airy_intensity_normalized

airy_intensity = airy(focal_length, wavenumber, radial_distance, aperture_radius)
    
#%% plotting, saving

fig, ax = plt.subplots(1,1, figsize = (3.5, 2.5))
ax.grid()

# rescale x axis to show um instead of m
radial_distance_microns = radial_distance * 1e6

# Plot tweezer theory result

ax.plot(radial_distance_microns,
        tweezer_intensity,
        label = r'$w_i \approx R$')

# Plot measurement result

ax.scatter(radial_distance_microns, measurement,
           label = 'measurement', 
           color = 'red', 
           s = 7,
           marker ='X'
           )

# Plot Airy theory result

ax.plot(radial_distance_microns, airy_intensity, label = 'point spread function')

ax.set_xlabel(r'$r$ [$\mu$m]', usetex = True)
ax.set_ylabel(r'$I/I_0$', usetex = True)
ax.legend()
ax.set_xlim(0, 2)

# Saving

plt.savefig('exports/AzimuthalAverage.pdf',
            dpi = 200,
            pad_inches = 0,
            bbox_inches = 'tight')
