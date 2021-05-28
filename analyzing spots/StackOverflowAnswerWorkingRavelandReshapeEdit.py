# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:20:51 2021

@author: Marijn Venderbosch
"""
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

# Loads image containing spot and ravels in order to fit
image = np.load('SpotCroppedNormalized.npy')
image_raveled = np.ravel(image)

def two_D_gaussian(X, amplitude, x0, y0, sigma_x, sigma_y):
    # We define the function to fit: a particular example of a 2D gaussian
    # indepdent variables x,y are passed as a single variable X (curve_fit only 
    # accepts 1D fitting, therefore the result is raveled 
    x, y = X
    x0 = float(x0)
    yo = float(y0)    
    exponent = -1 / (2 * sigma_x)**2 * (x - x0)**2 + -1 / (2 * sigma_y)**2 * (y - y0)**2
    I = amplitude * np.exp(exponent)
    return I.ravel()

# Amount of data points = pixels of the image
x_pixels = image.shape[0]
y_pixels = image.shape[1]

# Create x and y indices. Create a grid using np.meshgrid()
x = np.arange(0, x_pixels, 1)
y = np.arange(0, y_pixels, 1)
x, y = np.meshgrid(x, y)

"""fitting"""
# Add initial guess to start fitting. Assume peak is in center of image
# Assume standard devations about 10% of image dimensions
initial_guess = (1, x_pixels / 2, y_pixels / 2, x_pixels / 10, y_pixels / 10)

# Fitting the data from the image using our twoDgaussian function
fit_parameters, covariance = optimize.curve_fit(two_D_gaussian, (x, y), image_raveled, p0 = initial_guess)
data_fitted = two_D_gaussian((x, y), *fit_parameters)

"""plotting"""
fig, ax = plt.subplots(1, 1)
ax.imshow(image, extent = (x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(x_pixels, y_pixels), 5, colors='w', linewidths = 1.5)
ax.set_xlabel('pixels x')
ax.set_ylabel('pixels y')
ax.set_title('Normalized intensity and 2D gaussian fit')
plt.savefig('FittedSpot.png', dpi = 500, tigh_layout = True)
plt.show()

