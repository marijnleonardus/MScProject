#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:41:07 2021

@author: Marijn Venderbosch

This script computes for a single spot
- azimuthal average
- 2D gaussian fit

And compares to result to diffraction theory

"""

#%% Imports

import numpy as np
import scipy.io
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.special import jv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize 
from matplotlib import cm

#%% Variables

#magnification from newport objective 
magnification = 67

# Camera pixel size
pixel_microns = 4.65

# Crop for fit
cropping_range = 5

# Threshold for LoG detection
threshold = 0.2

# Number of spots expected for LoG. Since we don't use a pattern of spots set to 1
number_spots_expected = 1

# mat file
location = 'data/'
file_name = 'bestone.mat'

# Parameters for theoretical plot
waist = 2e-3
aperture_radius = 2e-3
wavelength = 820e-9
wavenumber = 2 * np.pi / wavelength
focal_length = 4e-3

# 2D plot zoom size
plot_range_x = 12
plot_range_y = 23

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
 

#%% Spot detection

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
        index += 1
    
    intensity_normalized = intensity_matrix / np.max(intensity_matrix)
    return intensity_normalized

measurement = azimuthal_average(image_cropped, maxima_locations)


#%% numerical integration, theory calculation

""" Integrate diffraction equation for gaussian waist w_i equal to aperture radius R"""

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
    
#%% Measurement vs result diffraction theory and tweezer Imshow

fig, (ax1, ax2) = plt.subplots(nrows = 1,
                               ncols = 2,
                               figsize = (7.9, 2.5))

# Imshow plot
def crop_center(img, cropx, cropy):

    startx = int(maxima_locations[0]) - cropx//2
    starty = int(maxima_locations[1]) - cropy//2    
    zoomed =  img[starty:starty+cropy, 
               startx:startx+cropx]
    return zoomed

image_zoomed = crop_center(image_cropped, 80, 60)

# Normalize
image_zoomed = image_zoomed / np.max(image_zoomed)

zoomed_plot = ax1.imshow(image_zoomed)
ax1.axis('off')

# Scalebar

# Define real life size of scalebar and corresponding amount of pixels
scalebar_object_size = 1e-6 #micron
scalebar_pixels = int(scalebar_object_size * magnification / pixel_microns * 1e6) # integer number pixels

scale_bar = AnchoredSizeBar(ax1.transData,
                           scalebar_pixels, # pixels
                           r'1 $\mu$m', # real life distance of scale bar
                           'upper left', 
                           pad = 1,
                           color = 'white',
                           frameon = False,
                           size_vertical = 1
                           )

ax1.add_artist(scale_bar)

# Colorbar

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right",
                          size = "8%",
                          pad = 0.1)
plt.colorbar(zoomed_plot, 
             cax=cax)


ax2.grid()
ax2.set_aspect(1.6)
ax2.tick_params(direction = 'in')

# rescale x axis to show um instead of m
radial_distance_microns = radial_distance * 1e6

# Plot tweezer theory result

ax2.plot(radial_distance_microns,
        tweezer_intensity,
        label = r'$w_i = R$')

# Plot measurement result
yerr = 0.03 
ax2.scatter(radial_distance_microns, measurement,
           label = 'measurement', 
           color = 'red', 
           s = 7,
           marker = 'X'
           )

# Plot Airy theory result
ax2.plot(radial_distance_microns, airy_intensity, label = r'$w_i \gg R$')
ax2.set_xlabel(r'$r$ [$\mu$m]', usetex = True)
ax2.set_ylabel(r'$I/I_0$', usetex = True)
ax2.legend()
ax2.set_xlim(0, 2)

# Annotate
ax1.annotate("(a)", xy = (0.05, 0.1), xycoords = "axes fraction", fontweight = 'bold', fontsize = 9, color = 'white')
ax2.annotate("(b)", xy = (0.05, 0.1), xycoords = "axes fraction", fontweight = 'bold', fontsize = 9)

plt.savefig('exports/AzimuthalAverageSpotZoomed.pdf',
            dpi = 200,
            pad_inches = 0,
            bbox_inches = 'tight')


#%% 2D fit

# Define 2D gaussian fit function. Formula is in thesis
def two_D_gaussian(X, amplitude, x0, y0, sigma_x, sigma_y):
    
    # We define the function to fit: a particular example of a 2D gaussian
    # indepdent variables x,y are passed as a single variable X (curve_fit only 
    # accepts 1D fitting, therefore the result is raveled 
    x, y = X 
    exponent = -0.5 * (sigma_x)**(-2) * (x - x0)**2 - 0.5 * (sigma_y)**(-2) * (y - y0)**2
    intensity = amplitude * np.exp(exponent)
    return intensity.ravel()

# Crop tweezer spot to RoI to fit. RoI is in the O(10) pixels
def crop_RoI(fit_image, maxima_locations, cropping_range):
    
    # Store coordinates where maximum was detected 
    row_max = int(maxima_locations[1])
    col_max = int(maxima_locations[0])
    
    # RoI boundaries. RoI is 2*cropping_range x 2*cropping_range
    lower_x = row_max - cropping_range
    upper_x = row_max + cropping_range + 1
    
    lower_y = col_max - cropping_range
    upper_y = col_max + cropping_range + 1
    
    # RoI crop
    fit_image = image_cropped[lower_x : upper_x,
                              lower_y : upper_y]
    
    # Normalize
    fit_image = fit_image / np.max(fit_image)
    
    # Ravel for fit
    fit_image_raveled = fit_image.ravel()
    return fit_image, fit_image_raveled
    
img_RoI, img_RoI_raveled = crop_RoI(image_cropped, maxima_locations, cropping_range)
 
# Fit needs 2D meshgrid
def twoD_mesh(cropping_range):
    
    # Pixel 1D matrices (discreet)

    pixels_x = np.arange(0, 2 * cropping_range + 1, 1)
    pixels_y = np.arange(0, 2 * cropping_range + 1, 1)
    
    # 2D matrix
    x, y = np.meshgrid(pixels_x, pixels_y)
    return x, y

pixels_mesh_x, pixels_mesh_y = twoD_mesh(cropping_range)

initial_guess = (1,  # amplitude
                 cropping_range, cropping_range, # center
                 cropping_range / 4, cropping_range / 4) # sigma

# Perform the 2D fit, using the Gaussian function and initial guess
popt, pcov = optimize.curve_fit(two_D_gaussian,
                                    (pixels_mesh_x, pixels_mesh_y),
                                    img_RoI_raveled,
                                    p0 = initial_guess)

# Store invididual fits 'popt' in a single variable containing all data over all the spots

fig3, ax3 = plt.subplots(figsize = (4, 4))

# Sigma radial direction
sigma_r_pixels = 0.5 * (popt[3] + popt[4])

#%% Plotting

# Plot images around (0,0) instead of origin in upper left corner.
extent = [-cropping_range ,cropping_range ,
          -cropping_range, cropping_range]

# Extend ensures axes go from - cropping_range to + cropping_range
ax3.imshow(img_RoI)
#ax3.set_axis_off()

# Plot circles with correct center and sigma. 
# Sigma is average of x and y, but also multiplied with 2 becaues its 1/e^2
circle_j = plt.Circle((popt[1] , 
                       popt[2] ), 
                      (popt[3] + popt[4]) ,
                      color = 'r', 
                      fill = False, 
                      linewidth = 1)
ax3.add_patch(circle_j)

# Plot crosses at center locations. Subtract cropping range to center (0,0)
# Radius is set to an arbitrarily small number, only its location is important
center_j = plt.Circle((popt[1] ,
                       popt[2]),
                      0.3,
                      color = 'r',
                      fill = True)
ax3.add_patch(center_j)

# We are interested in the quality of the fit: the R_squared. 
def R_squared(pixels_mesh_x, pixels_mesh_y, cropping_range, image):
    
    # residuals = ydata - f(xdata, *popt) where popt are fit 
    # We reshape the output of the 2D gaussian to a square array
    residuals = img_RoI - two_D_gaussian((pixels_mesh_x, pixels_mesh_y), *popt).reshape(2 * cropping_range + 1, 2 * cropping_range + 1)
    
    # ss_res is the sum over all invididual residuals, square to keep positive numbers
    ss_res = np.sum(residuals**2)
    # Total sum of squares is the sum over (ydata-mean(ydata))^2
    ss_tot = np.sum((img_RoI - np.mean(img_RoI))**2)
    
    # Definition of the R^2
    r_squared=1- (ss_res/ ss_tot)
    return r_squared

r_squared = R_squared(pixels_mesh_x, pixels_mesh_y, cropping_range, img_RoI)

"""plot 3D fit result in 3D plot"""

fig =plt.figure(figsize = (5, 4))
ax = plt.axes(projection='3d')

# Plot data from camera as dots
ax.scatter3D(pixels_mesh_x, pixels_mesh_y, img_RoI,
             color = 'black',
             s = 1,
             label = 'Data points',
             cmap = cm.jet
             )

# Plot gaussian fit 
peak = two_D_gaussian((pixels_mesh_x, pixels_mesh_y), *popt).reshape(2 * cropping_range + 1, 2 * cropping_range + 1)
im = ax.plot_surface(pixels_mesh_x, pixels_mesh_y, peak,
                rstride = 1,
                cstride = 1,
                alpha = 0.5,
                label = '2D Gaussian fit',
                cmap = cm.jet
                )

ax.invert_xaxis()
ax.tick_params(axis='x', which='major', pad=-2)
ax.tick_params(axis='y', which='major', pad=-2)
#ax.invert_yaxis()

ax.set_xlabel(r'$x$ [pixels]',
              labelpad = -2,
              usetex = True)

ax.set_ylabel(r'$y$ [pixels]', 
              labelpad = -2,
              usetex = True)

ax.set_zlabel(r'$G(x,y)/G_0$', 
              labelpad = -1,
              usetex = True)

ax.view_init(20, 40)

#%% Saving, printing

print("R-squared is: " + str(r_squared))

waist = 2 * sigma_r_pixels * 4.65 / magnification
print("Waist is: " + str(waist))

plt.savefig('exports/3DSpotFitGaussianSmaller.pdf', 
            dpi = 200, 
            pad_inches = 0,
            bbox_inches = 'tight')