# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:19:30 2021
@author: Marijn Venderbosch
This script
    - Takes raw camera data and saves region of interest
    - Detects spot using the laplacian of Gaussian algorithm
    - Crops regions around spots
    - Fits 2D Gaussians on spots
    - Spots histograms of beam widths and trap depths 
"""

#%% Imports

import numpy as np
import scipy.io
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from scipy import optimize 
from scipy.stats import norm
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

#%% Variables

# Number of spots that we make, to check if spot detection worked
number_spots_expected = 9
# Location of .mat data file
mat_file_location = 'files/10 4 adjusted spacing/3x3.mat'
# Threshold on how sensitive spot detection is
threshold = 0.2
# How many pixels do we crop around the spot maxima locations
cropping_range = 12
# magnification from newport objective. This is uncalibrated. 
magnification = 67
pixel_size_microns = 4.65

#%% Load .mat 

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
    #cam_frame_cropped = cam_frame[cam_x_min : cam_x_max , cam_y_min : cam_y_max]
    cam_frame_cropped = cam_frame[200 : 800, 50 : 650]
    
    # Save cropped frame as numpy array
    np.save('files/cam_frame_array_cropped', cam_frame_cropped)
 
# execute function. Insert in brackets the .mat filename
load_and_save(mat_file_location)


#%% Spot detection using LoG algorithm

# Load image from script 'loadmatSaveCamFrame.py'. 
# Image is cropped to region of interest
# Transpose because camera is rotated
image_transposed = np.load('files/cam_frame_array_cropped.npy')
image = np.transpose(image_transposed)

# Show camera image
image_normalized = image / np.max(image)

# plot for camera image and LoG spots
fig, (axCam,axLoG) = plt.subplots(ncols = 2,
                                  nrows = 1, 
                                  sharey = True,
                                  tight_layout = True,
                                  figsize = (7.7, 3.4))

axCam.imshow(image_normalized, 
             cmap = 'gray'
          #cmap = 'gray'
          )    
   
axCam.set_xlabel(r'$x$ [pixels]')
axCam.set_ylabel(r'$y$ [pixels]')
    
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
factor = 6
sizes = spots_LoG[:, 2] * factor


# Plot original image and overlay with crosses on spots where blobs are detected
# Radii or cicles are from the gaussian kernels that detected them
axLoG.set_xlabel(r'$x$ [pixels]')

axLoG.imshow(image,
            cmap = 'gray'
            )

axLoG.scatter(maxima_x_coordinates , maxima_y_coordinates,
             marker = 'x',
             s = sizes, 
             color = 'w', 
             linewidth = 0.7
             )

# Annotate
axCam.text(-110,
         30,
         r'a)',
         fontsize = 14,
         fontweight = 'bold'
         )
axLoG.text(-70,
         30,
         r'b)',
         fontsize = 14,
         fontweight = 'bold'
         )

# colorbar
#fig.colorbar(cm.ScalarMappable(cmap = 'viridis'),
#             ax = axLoG)

# Saving and showing
plt.savefig('exports/CamImgLoGSpots.pdf', 
            pad_inches = 0,
            dpi = 400,
            bbox_inches = 'tight'
            )

#%% Crop spots and fit gaussians

"""This script crops the spots around the locatins found by the LoG algorithm. 
Subsequently it fits 2D gaussians around the spot locations and plots them
"""

# Import image as well as spot locations as found by LoG
image = np.load('files/cam_frame_array_cropped.npy')
spot_locations = np.load('files/spots_LoG.npy')

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

# Normalization
maximum_spot_intensity = np.max(image)

# Make an empty list, each entry being an empty array 
spots_cropped = [0] * amount_spots

# Crop around each individual spot and store in the list
for k in range(amount_spots):
    spots_cropped[k] = image[lower_limit_x[k]:upper_limit_x[k] ,
                             lower_limit_y[k]:upper_limit_y[k]]
    # normalize
    spots_cropped[k] = spots_cropped[k] / maximum_spot_intensity    
 
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
    
    exponent = -0.5 * (sigma_x)**(-2) * (x - x0)**2 - 0.5 * (sigma_y)**(-2) * (y - y0)**2
    function = amplitude * np.exp(exponent)
    return function.ravel()

# Initial values. The fitting algorithm needs an initial guess. Esimated from 
# plot of the spot. 
initial_guess = (1, cropping_range, cropping_range, cropping_range / 3, cropping_range / 3)

# In the for loop, every iteration we want to store data
# We need to initialize empty lists to store these variables. All variables are intially the
# same empty list with dimensions equal to the amount of spots. 
spot_raveled = max_Gauss_locations_list = [0] * amount_spots

# Initialize plot o
fig, axes = plt.subplots(amount_subplots, amount_subplots, figsize = (5, 6), sharex = True, sharey = True)
fig.suptitle('Spots Cropped Around Maxima (pixels)')

# To be able to sum over axes it needs to be raveled
if number_spots_expected ==1:
    ax = axes
else: 
    ax = axes.ravel()

# Initilize empty list to fit all paramters in. Because it is appended each iteration, it
# can start empty
fit_parameters = []
trapdepth_list = []
sigma_list = []
r_squared_list = []

# For each picture do a 2D Gaussian fit and plot them
for j in range(amount_spots):
    # the images containing the spots need to be raveled 
    # the 2D fit can only iterate over one direction
    spot_raveled[j] = spots_cropped[j].ravel()
    
    # Perform the 2D fit, using the Gaussian function and initial guess
    popt, pcov = optimize.curve_fit(two_D_gaussian,
                                    (x, y),
                                    spot_raveled[j],
                                    p0 = initial_guess)
    
    # Store invididual fits 'popt' in a single variable containing all data over all the spots
    fit_parameters.append(popt)
    # Store in single variable containing all data: fit_parameters
    # Store sigma, trap depth as well as max. locations
    sigma_r = 0.5 * (popt[3] + popt[4])
    sigma_list.append(sigma_r)
    trapdepth_list.append(popt[0])
    # Store peak middle locations. These are deviations from the LoG locations (sub-pixel)
    max_Gauss_locations_list[j] = [popt[1] - cropping_range,
                                   popt[2] - cropping_range]
    
    # Plotting
    # Plot images around (0,0) instead of origin in upper left corner. 
    extent = [-cropping_range ,cropping_range ,
              -cropping_range, cropping_range]
    
    # Extend ensures axes go from - cropping_range to + cropping_range
    ax[j].imshow(spots_cropped[j], extent = extent)
    # Title: index but starting from 1 instead of 0 so add 1
    ax[j].set_title('m ='+ str(j + 1))
    ax[j].set_axis_off()
    
    # Plot circles with correct center and sigma. 
    # Sigma is average of x and y, but also multiplied with 2 becaues its 1/e^2
    circle_j = plt.Circle((popt[1] - cropping_range, popt[2] - cropping_range), (popt[3] + popt[4]) , color = 'r', fill = False, linewidth = 1)
    ax[j].add_patch(circle_j)
    
    # Plot crosses at center locations. Subtract cropping range to center (0,0)
    # Radius is set to an arbitrarily small number, only its location is important
    center_j = plt.Circle((popt[1] - cropping_range,
                           popt[2]- cropping_range),
                          0.3,
                          color = 'r',
                          fill = True)
    ax[j].add_patch(center_j)
    
    # We are interested in the quality of the fit: the R_squared. 
    # residuals = ydata - f(xdata, *popt) where popt are fit 
    # We reshape the output of the 2D gaussian to a square array
    residuals = spots_cropped[j] - two_D_gaussian((x,y), *popt).reshape(2*cropping_range + 1, 2*cropping_range + 1)
    # ss_res is the sum over all invididual residuals, square to keep positive numbers
    ss_res = np.sum(residuals**2)
    # Total sum of squares is the sum over (ydata-mean(ydata))^2
    ss_tot = np.sum((spots_cropped[j] - np.mean(spots_cropped[j]))**2)
    # Definition of the R^2
    r_squared_list.append(1- (ss_res/ ss_tot))
    
# Because we used lists to append, we need to convert to numpy arrays
sigma_matrix = np.array(sigma_list)
trapdepth_matrix = np.array(trapdepth_list)
r_squared_matrix = np.array(r_squared_list)

# Find average and spread in R^2
mu_r_squared, stddev_r_squared = norm.fit(r_squared_matrix)

# Print result for convenience
print("Average r^2 is: " + str(mu_r_squared))

# Saving and showing    
plt.savefig('exports/SpotsCropped_range10.png', 
            pad_inches = 0,
            dpi = 500)

# #%% 3D G(x,y) fit of only one spot

# # Initialize figure
# fig =plt.figure(figsize = (5, 4))
# ax = plt.axes(projection='3d')

# # Plot data from camera as dots
# ax.scatter3D(x,y,spots_cropped[0],
#               color = 'black',
#               s = 1,
#               label = 'Data points'
#               )

# # Plot gaussian fit 
# first_peak_parameters = fit_parameters[0]
# first_peak = two_D_gaussian((x,y),*first_peak_parameters).reshape(2*cropping_range+1,2*cropping_range+1)
# im = ax.plot_surface(x,y,first_peak,
#                 rstride = 1,
#                 cstride = 1,
#                 alpha = 0.5,
#                 cmap = cm.jet,
#                 label = '2D Gaussian fit'
#                 )

# ax.invert_xaxis()
# ax.tick_params(axis='x', which='major', pad=-2)
# ax.tick_params(axis='y', which='major', pad=-2)
# #ax.invert_yaxis()

# ax.set_xlabel(r'$x$ [pixels]',
#               labelpad = -2,
#               usetex = True)

# ax.set_ylabel(r'$y$ [pixels]', 
#               labelpad = -2,
#               usetex = True)

# ax.set_zlabel(r'$G(x,y)/G_0$', 
#               labelpad = -1,
#               usetex = True)

# ax.view_init(20, 35)

# plt.savefig('exports/3DSpotFitGaussian.pdf', 
#             dpi = 200, 
#             pad_inches = 0)

# We want to store the exact spot locations in (pixels_x, pixels_y), but sub-pixel from fits
def store_max_peaks_subpixel(list_input):
    # Convert to numpy array first from list
    max_Gauss_locations_array = np.array(list_input)

    # Store x,y subpixel locations
    max_Gauss_locations_subpixels_x = maxima_x_coordinates + max_Gauss_locations_array[:, 0]
    max_Gauss_locations_subpixels_y = maxima_y_coordinates + max_Gauss_locations_array[:, 1]

    # Store in single array
    max_Gauss_locations_subpixels = np.column_stack((max_Gauss_locations_subpixels_x, max_Gauss_locations_subpixels_y))
    return max_Gauss_locations_subpixels

# Call function and store result in variable
max_Gauss_locations_subpixels = store_max_peaks_subpixel(max_Gauss_locations_list)

#%% Spacing calculation: calculate spacing in x and y

# Dimension: array is d x d where d is dimension
dimension = int(np.sqrt(amount_spots))

def spacing_calculator(locs):
    # Initialize empty matices. xdif is initially longer but later stuff will be omitted          

    x_spacing_overfill = np.zeros(amount_spots - 1)
    y_spacing = np.zeros(amount_spots - dimension)
    
    # Compute euclidean space between iterative points
    for k in range(amount_spots - 1):
        x_spacing_overfill[k] = np.sqrt((locs[k + 1, 0] - locs[k, 0])**2 + (locs[k + 1, 1] - locs[k, 1])**2)
    
    # x_spacing_overfill contains too many rows. As we jump to the next row in the array the calculation doesnt
    # make sense. For a 3x3 array e.g. we remove row 2, 5, 8 using np.delete()
    x_spacing = np.delete(x_spacing_overfill, list(range(2, x_spacing_overfill.shape[0], 3)), axis = 0)
        
    # y_spacing does contain the right amount, we just skip the calculation over the last row
    # So we decrease the stepper by the dimension size
    for k in range(amount_spots - dimension):
        y_spacing[k] = np.sqrt((locs[k + 3, 0] - locs[k, 0])**2 + (locs[k + 3, 1] - locs[k, 1])**2)
        
    # convert magnification to physical size
    x_spacing_microns = x_spacing * pixel_size_microns  / magnification
    y_spacing_microns = y_spacing * pixel_size_microns / magnification
    
    return x_spacing_microns, y_spacing_microns
        
# Call function. Save result in arrays x_spacing and y_spacing respectively 
x_spacing_microns, y_spacing_microns = spacing_calculator(max_Gauss_locations_subpixels)

# Calculate average and spread in spacings
mu_x_spacing_microns, stddev_x_spacing_microns = norm.fit(x_spacing_microns)
mu_y_spacing_microns, stddev_y_spacing_microns = norm.fit(y_spacing_microns)  

# Calculate ratio between x and y spacing 
ratio_x_y_spacing = mu_y_spacing_microns / mu_x_spacing_microns   

#%% Histograms of distributions
"""The following script will plot histograms of the obtained beamwidths and 
trap depths, as well as finding the averages and spreads in them."""

# compute beam widths
beam_width_pixels = 2 * sigma_matrix 

# pixels are 4.65 micron. Magnification onto camera is 60X
# 2*sigma corresponds to the 1/e^2 radius
beamwidth_microns = beam_width_pixels * pixel_size_microns / magnification

# Obtain average and spreads in beamwidth and trapdepth by fitting data with a Gaussian
mu_beam_width, stddev_beam_width = norm.fit(beamwidth_microns)
mu_trap_depth, stddev_trap_depth = norm.fit(trapdepth_matrix)

# A problem is that the trap depth is now normalized to the highest value it achieves
# So we once again 'normalize' such that the average is one. We call this 'unity' 
trapdepth_matrix_unity = trapdepth_matrix / mu_trap_depth
mu_trap_depth_unity = mu_trap_depth / mu_trap_depth
stddev_trap_depth_unity = stddev_trap_depth / mu_trap_depth

# Histogram plots

# Number of bins: increase for more spots. Takes square root of number of spots and
# rounds to nearest integer.
n_bins = int(np.sqrt(amount_spots))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6.5, 3.8))


# Plot histograms: normalized using 'density' option
ax1.hist(beamwidth_microns, 
         bins = 9, 
         #hatch = '/', 
         alpha = 0.7,
         density = True,
         edgecolor = 'black',
         label = 'histogram'
         )

ax2.hist(trapdepth_matrix_unity, 
         bins = n_bins,
         #hatch = '/',
         alpha = 0.7,
         edgecolor = 'black',
         density = True
         )

# Get same limits for Gausian as plot range
xmin_beamwidth, xmax_beamwidth = ax1.get_xlim()
xmin_trapdepth, xmax_trapdepth = ax2.get_xlim()

# Get a stepsize for the Normal distribution. 10^2 should do
number_steps = 100
x_beamwidth = np.linspace(xmin_beamwidth, xmax_beamwidth, number_steps)
x_trapdepth = np.linspace(xmin_trapdepth, xmax_trapdepth, number_steps)

# Generate the normal distribution data using the norm.pdf function from scipy.stat
normal_distribution_beamwidth = norm.pdf(x_beamwidth, mu_beam_width, stddev_beam_width)
normal_distribution_trapdepth = norm.pdf(x_trapdepth, mu_trap_depth_unity, stddev_trap_depth_unity)

# We want to use the result in the titles. But there are too many digits. 
# Round to 2 significant digits.
# Print result for easy verify of correctness
beamwidth_final =  "%0.*f"%(2 , mu_beam_width)
beamwidth_final_spread = "%0.*f"%(2 , stddev_beam_width)
print("1/e^2 radius is: "+ str(beamwidth_final))

trapdepth_final = "%0.*f"%(2 , mu_trap_depth_unity)
trapdepth_spread = "%0.*f"%(2 , stddev_trap_depth_unity)

# Print relative error, multiply times 100 for percentage
trapdepth_spread_relative = "%0.*f"%(1, 100* stddev_trap_depth_unity / mu_trap_depth_unity)
print("Trap depth relative error is: "+ str(trapdepth_spread_relative)+"%")

# Plot the normal distributions. 
ax1.plot(x_beamwidth, normal_distribution_beamwidth,
         'r--',
         linewidth = 2,
         alpha = 0.8,
         label = 'normal distribution'
         )

ax2.plot(x_trapdepth, normal_distribution_trapdepth, 
         'r--',
         alpha = 0.8,
         linewidth = 2)

# Edit labels
ax1.set_xlabel(r'$w_0$ $(1/e^2$ radius) [$\mu$m]', usetex = True)
ax1.set_yticklabels([])
ax1.xaxis.set_major_locator(MultipleLocator(0.05))
ax1.xaxis.set_minor_locator(MultipleLocator(0.01))
ax1.yaxis.set_major_locator(MultipleLocator(1.69))
ax1.set_ylabel('Counts [a.u.]')

ax2.set_xlabel(r'$U_0/\left\langle U_0 \right\rangle$', usetex = True)
ax2.set_yticklabels([])
ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.yaxis.set_major_locator(MultipleLocator(0.5))

# Annotate
ax1.text(0.905,
         13,
         r'$(0.89\pm0.03)$ $\mu$m',
         fontsize = 9,
         color = 'r'
         )
ax2.text(1.055,
         13 *.5/1.69,
         r'$(1.00\pm0.10)$',
         fontsize = 9,
         color = 'r'
         )

fig.legend(loc='upper left', bbox_to_anchor=(0.115, 1.03))

# Save plot
plt.savefig('exports/FittedHistograms.pdf', 
            dpi = 300,
            pad_inches = 0,
            bbox_inches = 'tight')

# Show all plots
plt.show()
