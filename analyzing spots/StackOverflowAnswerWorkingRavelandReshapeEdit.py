# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:20:51 2021

@author: Marijn Venderbosch
"""
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

def twoDgaussian(X, amplitude, x0, y0, sigma_x, sigma_y):
    # We define the function to fit: a particular example of a 2D gaussian
    # indepdent variables x,y are passed as a single variable (curve_fit only) 
    # accepts 1D fitting, therefore the result is raveled 
    x, y = X
    x0 = float(x0)
    yo = float(y0)    
    exponent = -1 / (2 * sigma_x)**2 * (x - x0)**2 + -1 / (2 * sigma_y)**2 * (y - y0)**2
    I = amplitude * np.exp(exponent)
    return I.ravel()

# Amount of data points = pixels of the image
x_pixels = 6
y_pixels = 6

# Create x and y indices. Create a grid using np.meshgrid()
x = np.arange(0, x_pixels, 1)
y = np.arange(0, y_pixels, 1)
x,y = np.meshgrid(x, y)

# Generate the data using the gaussian function and the generated grid
data = twoDgaussian((x, y), 1, x_pixels / 2, y_pixels/2, x_pixels / 10, y_pixels / 20)

# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data.reshape(x_pixels ,y_pixels ))
plt.colorbar()

# add some noise to the data and try to fit the data generated beforehand
initial_guess = (1, x_pixels/2,y_pixels/2, x_pixels / 10, y_pixels / 20)

data_noisy = data + 0.05 * np.random.normal(size = data.shape)

popt, pcov = opt.curve_fit(twoDgaussian, (x, y), data_noisy, p0 = initial_guess)

data_fitted = twoDgaussian((x, y), *popt)

fig, ax = plt.subplots(1, 1)
ax.imshow(data_noisy.reshape(x_pixels, y_pixels), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(x_pixels, y_pixels), 8, colors='w')
plt.show()
