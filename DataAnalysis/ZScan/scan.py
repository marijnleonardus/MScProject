#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep  1 12:21:11 2021

@author: Marijn Venderbosch

This script plots a 3d scan of the tweezer. 
Dimensions are set using the scanning step size in z-direction
and pixel size in the r direction

Data is compared to theory result from 'defocus longitudinal' script
"""

#%% Imports

import numpy as np
import scipy.io
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy.optimize

#%% Variables

# mat file location
mat_files_location = "./data/0_1astigcorrection0_27spherical/"

lam = 810e-9
k = 2 * np.pi / lam
f = 4e-3
R = 2e-3
w_i = 2e-3
magnification = 60
pixel_size = 4.65
threshold = 0.05
# amount of space to see around the maximum location
number_spots_expected = 1
row_cropping_range = 35

# z scan names of .mat files to import
z_steps_per_image = 50
z_start = -800
z_stop = 0
step = 0.010585

# plot range theory result
plot_range = 7e-6
dz = np.linspace(-plot_range, plot_range, 1000)

# Only fit around waist, in microns
fitting_range = 5

#%% Functions

# Diffraction theory axial diffraction pattern 

def intensity_defocus(u):
    
    # in terms of dimenionless defocus paramter u = k dz R**2/f**2
    u = k * dz * R**2 / f**2

    intensity_defocus = (-2 * np.exp(1) * np.cos(u / 2) + np.exp(2) + 1) / (np.exp(2)* (u**2 + 4))
    intensity__defocus_normalized = intensity_defocus / np.max(intensity_defocus)
    
    return intensity__defocus_normalized

# Rescale x axis
dz_microns = dz * 1e6

intensity__defocus_normalized = intensity_defocus(dz_microns)


# Spot detection

def spot_detection(image):
    spots_LoG = blob_log(image,
                         max_sigma = 30, 
                         num_sigma = 10, 
                         threshold = threshold)

    # Check if expected amount of spots is detected
    number_spots_found = spots_LoG.shape[0]
    if number_spots_expected != number_spots_found:
        print('Error: spot finding did not find the expected number of spots')

    # Compute radii in the 3rd column by multipling with sqrt(2)
    spots_LoG[:, 2] = spots_LoG[:, 2] * np.sqrt(2)

    # Find maxima locations and sizes. x and y are swapped becaues tranposed
    maxima_y_coordinates = spots_LoG[:, 0]
    maxima_x_coordinates = spots_LoG[:, 1]
    
    # return for next script
    maximum_locs = np.array([maxima_x_coordinates, maxima_y_coordinates])
    return maximum_locs

# Loads file, crops to defined RoI

def load_crop_RoI_normalize(mat_files_location, filename_list):
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
    cam_frame = np.transpose(full_frame[cam_x_min : cam_x_max , 
                                        cam_y_min : cam_y_max])
    
    return cam_frame

# Fit Gaussian to obtain amplitudes

fit_radial_coordinate = np.linspace(-2, 2, 100)

def gaussian_1D(x_data, amplitude, center, sigma):

    exponent = -0.5 * ((x_data - center) / sigma)**2
    gaussian = amplitude * np.exp(exponent)
    
    return gaussian


#%% Prepping and executing for loop that loops over scanned images

# assign dataset names
filename_list = list(map(str, 
                         np.arange(z_start, z_stop + z_steps_per_image, z_steps_per_image).tolist()
                         ))

# Create empty lists to later store data from for loop in
cam_frames = []
rows = []
longitudinal_profile = []
fit_parameters_list = []

# Fit parameter empty matrix
fit_parameters = np.zeros([17, 3])

# Fit guess: offset, amplitude, center, sigma
guess = [170, row_cropping_range, 5]
row_x_data = np.linspace(0, 2 * row_cropping_range, 2 * row_cropping_range)

# Execute for loop
# Each iteration, the results are saved in the empty matrices defined above

for i in range(len(filename_list)):
    
    # Use RoI crop function defined above
    cam_frame = load_crop_RoI_normalize(mat_files_location, filename_list)
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
    
    # Fit each picture to obtain amplitude, this might be more accurate than simply
    # Taking the maximum value
    fit_parameters, fit_covariance = scipy.optimize.curve_fit(gaussian_1D,
                                                             row_x_data,
                                                             row_cropped,
                                                             p0 = guess)
    fit_parameters_list.append(fit_parameters)
        
    # Save longitudal intensity
    longitudinal = np.max(cam_frame)
    longitudinal_profile.append(longitudinal)

# Fitted values

fit_matrix = np.array(fit_parameters_list)
amplitudes_matrix = fit_matrix[:, 0]
amplitudes_matrix_normalized = amplitudes_matrix / np.max(amplitudes_matrix)

fit_centra = fit_matrix[:, 1]
    

# Convert list of rows to array (2D array: r,z)    
array = np.transpose(np.array(rows))  

# Normalize
array = array / np.max(array)

# Compute aspect ratio for plot

z_dimension = len(filename_list) * z_steps_per_image * step
r_dimension = (2 * row_cropping_range + 1) * pixel_size / magnification
aspect_ratio = r_dimension / z_dimension

# Normalize longitudinal data
longitudinal_normalized = longitudinal_profile / np.max(longitudinal_profile)

# Make x axis for longitudinal plot
x_longitudinal = np.linspace(-z_dimension / 2,
                             z_dimension / 2,
                             len(filename_list))

#%% fitting

# Fit longituninal_profile
def gaussian(x, amplitude, x0, sigma):
    
    # We define a gaussian to fit the data
    exponent = -0.5 * (sigma)**(-2) * (x - x0)**2 
    
    intensity = amplitude * np.exp(exponent)
    return intensity

def GaussianAxial(z, amplitude, z0, zR):
    intensity = amplitude / (1 + (z - z0)**2 / zR**2)
    return intensity

# Fitting x variable, more accurate
fit_x = np.linspace(-2 / 3 * z_dimension, 2 / 3 * z_dimension, 100)

# Fit limited range. Fitting range in um
sliced_indices = np.where(
    (x_longitudinal <= fitting_range) & (x_longitudinal >= -fitting_range)
    )

# Fitting, we don't fit the entire plot, just the region around the waist

x_longitudinal_cropped = np.array(x_longitudinal[sliced_indices])
longitudinal_normalized_cropped = np.array(longitudinal_normalized[sliced_indices])

# Initial guess for fitting; amplitude, center, sigma

fitting_guess = [1, -0.4, 4] 
# Fitting limited plot range.
# Fits gaussian beam which in axial direction goes as U0/(1 + z^2/zR^2)
poptAxial, pcovAxial = scipy.optimize.curve_fit(GaussianAxial,
                                                 x_longitudinal_cropped,
                                                 longitudinal_normalized_cropped, 
                                                 p0 = fitting_guess
                                                )

center_x_fit = poptAxial[1]

gaussianAxialFit = GaussianAxial(fit_x, *poptAxial)

#%% First and second plot: Slice of tweezer
 
fig, (ax1 ,ax2) = plt.subplots(nrows = 2, 
                              ncols = 1, 
                              figsize = (3.5, 3.5*7/5),
                              sharex = True)
fig.subplots_adjust(hspace = 0, wspace = 0)

# Vertical distance between plots
plt.subplots_adjust(hspace = 0)

"""first plot: the tweezer scan"""

# The aspect ratio of the plot is fixed to calculated value to ensure spacing in r,z is the same

ax1.set_aspect(aspect_ratio)

# The aspect ratio of x,y axis is set to the same value
ax1.imshow(array,
          aspect = 1,
          extent = [-z_dimension/2 - center_x_fit, z_dimension/2 - center_x_fit,
                    -r_dimension/2,r_dimension/2
              ])

# Ticks, labels first plot

ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.set_ylabel(r'$r$ [$\mu$m]', usetex = True)

# Setting horizontal plot range
#ax1.set_xlim(-4, 4)

"""second plot: the data for the tweezer scan"""

# Longitudinal plot, second plot
# Plot against same horizontal coordinate

ax2.grid()
ax2.errorbar(x_longitudinal - center_x_fit, amplitudes_matrix_normalized, 
            color = 'black',
            fmt = 'o',
            ms = 3,
            yerr = 0.1 * amplitudes_matrix_normalized
            )

# Plots, ticks for second plot

ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax2.set_xlabel(r'$\delta z$ [$\mu$m]', usetex = True)
ax2.set_ylabel(r'$I/I_0$', usetex = True)

# only labels on bottom plot

for ax in fig.get_axes():
    ax.label_outer()
    
ax2.plot(dz_microns, intensity__defocus_normalized)
ax2.set_xlim(-4.2, 4.2)

# Saving

plt.savefig('exports/AxialImageTweezerScan.pdf',
            dpi = 200,
            pad_inches = 0,
            bbox_inches = 'tight')

#%% Second plot: fit plot

"""third plot. We plot separately because x range is different"""
fig, ax = plt.subplots(figsize = (4, 3))
ax.errorbar(x_longitudinal_cropped, longitudinal_normalized_cropped, 
            color = 'blue',
            fmt = 'o',
            ms = 5, 
            yerr = 0.05 * longitudinal_normalized_cropped
            )

ax.plot(fit_x, gaussianAxialFit, color = 'red')
ax.set_xlim(-fitting_range, +fitting_range)

ax.set_xlabel(r'$\delta z$ [$\mu$m]', usetex = True)
ax.set_ylabel(r'$I/I_0$', usetex = True)
ax.grid()

# Saving

plt.savefig('exports/FittedRayleigh.pdf',
            dpi = 200,
            pad_inches = 0,
            bbox_inches = 'tight')

rayleigh = poptAxial[2]

print('Fitted rayleigh: ' + str(rayleigh))
