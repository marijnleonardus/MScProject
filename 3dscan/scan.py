# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:21:11 2021

@author: Marijn Venderbosch

This script plots a 3d scan of the tweezer. 
Dimensions are set using the scanning step size in z-direction
and pixel size in the r direction

Compare to theory result from 'defocus longitudinal' script
"""

#%% Imports, variables
import numpy as np
import scipy.io
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy.optimize

# mat file location
mat_files_location = "./files/01/"

# variables
lam = 780e-9
k = 2 * np.pi / lam
f = 4e-3
R = 2e-3
w_i = 2e-3
magnification = 71.14
pixel_size = 4.65
threshold = 0.05
# amount of space to see around the maximum location
number_spots_expected = 1
row_cropping_range = 35

# z scan names of .mat files to import
z_steps_per_image = 50
z_start = -850
z_stop = 0
step = 0.00934

# plot range theory result
plot_range = 15e-6
dz = np.linspace(-plot_range, plot_range, 1000)

# Only fit around waist, in microns
fitting_range = 2.5

#%% Mathemetica result and PSF, theory results
def intensity_defocus(u):
    # in terms of dimenionless defocus paramter u = k dz R**2/f**2
    u = k * dz * R**2 / f**2

    intensity_defocus = (-2 * np.exp(1) * np.cos(u / 2) + np.exp(2) + 1) / (np.exp(2)* (u**2 + 4))
    intensity__defocus_normalized = intensity_defocus / np.max(intensity_defocus)
    
    return intensity__defocus_normalized
# Rescale x axis
dz_microns = dz * 10e5

intensity__defocus_normalized = intensity_defocus(dz_microns)


#%% function spot detection
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
longitudinal_profile = []

#%% make 'sideview' image by sticking together individual camera images
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
    
    # Save longitudal intensity
    longitudinal = np.max(cam_frame)
    longitudinal_profile.append(longitudinal)
    

# Convert list of rows to array (2D array: r,z)    
array = np.transpose(np.array(rows))  

# Compute aspect ratio for plot
# Z dire
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

# Fitting x variable, more accurate
fit_x = np.linspace(-2 / 3 * z_dimension, 2 / 3 * z_dimension, 100)

# Fit limited range. Fitting range in um
sliced_indices = np.where(
    (x_longitudinal <= fitting_range) & (x_longitudinal >= -fitting_range)
    )
# Fitting, we don't fit the entire plot, just the region around the waist
x_longitudinal_cropped = np.array(x_longitudinal[sliced_indices])
longitudinal_normalized_cropped = np.array(longitudinal_normalized[sliced_indices])

fitting_guess = [1, -0.4, 4] #amplitude, center, sigma
# Fitting limited plot range
popt, pcov = scipy.optimize.curve_fit(gaussian, x_longitudinal_cropped, longitudinal_normalized_cropped, p0 = fitting_guess)

center_x_fit = popt[1]

gaussian_fitted = gaussian(fit_x, *popt)

#%% plotting
# Initialize plot
fig, (ax1 ,ax2) = plt.subplots(nrows = 2, 
                              ncols = 1, 
                              figsize = (6,8),
                              sharex = True)

# Vertical distance between plots
plt.subplots_adjust(hspace = 0.5)

"""first plot: the tweezer scan"""
# The aspect ratio of the plot is fixed to calculated value to ensure spacing in r,z is the same
ax1.set_aspect(aspect_ratio)
# The aspect ratio of x,y axis is set to the same value
ax1.imshow(array,
          aspect = 1,
          extent = [-z_dimension/2, z_dimension/2,
                    -r_dimension/2,r_dimension/2
              ])

# Ticks, labels first plot
ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.set_ylabel(r'Radial direction [$\mu$m]')

# Setting horizontal plot range
ax1.set_xlim(-4, 4)

"""second plot: the data for the tweezer scan"""
# Longitudinal plot, second plot
# Plot against same horizontal coordinate
ax2.grid()
ax2.scatter(x_longitudinal - center_x_fit, longitudinal_normalized, 
            color = 'blue',
            s = 6, 
            marker = 'X'
            )

# Plots, ticks for second plot
ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax2.set_xlabel(r'z-direction [$\mu$m]')
ax2.set_ylabel('Irradiance [a.u.]')

# only labels on bottom plot
for ax in fig.get_axes():
    ax.label_outer()
    
ax2.plot(dz_microns, intensity__defocus_normalized)
ax2.set_xlim(-4, 4)

fig.tight_layout()

# Saving
plt.savefig('exports/4mmFScorrection01.pdf',
            dpi = 300)



"""third plot. We plot separately because x range is different"""
fig, ax = plt.subplots()
ax.scatter(x_longitudinal_cropped, longitudinal_normalized_cropped, 
            color = 'blue',
            s = 6, 
            marker = 'X'
            )

ax.plot(fit_x, gaussian_fitted)
ax.set_xlim(-fitting_range, +fitting_range)

ax.set_xlabel(r'Defocus [$\mu$m]')
ax.set_ylabel(r'Intensity [a.u.]')
ax.grid()



#%% Saving
plt.savefig('exports/4mmFScorrection01_lim_fitrange.pdf',
            dpi = 300)

waist = 2* popt[2]
rayleigh = np.sqrt(2 * np.log(2)) * popt[2]

print('Fitted waist: '+ str(waist))
print('Fitted rayleigh: ' + str(rayleigh))
