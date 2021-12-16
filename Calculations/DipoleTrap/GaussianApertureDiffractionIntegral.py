#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Aug 23 23:39:23 2021

@author: marijn venderbosch

Script computes diffraction of Gaussian beam with through aperture with radius R
The waist (1/e² radius) of this aperture is assumed to be equal to the aperture radius, though
this does not have to be the case
"""

#%% Imports

import numpy as np
import scipy.integrate
from  scipy.special import jv
import matplotlib.pyplot as plt

#%% Variables

waist = 2e-3 #m
aperture_radius = 2e-3 #m
wavelength = 820e-9 #m
wavenumber = 2 * np.pi / wavelength #1/m
eq_focal_length = 4e-3 # m

#%% Calculations

# Diffraction integral is computed as a function of the radial coordinate in focal plane: r'
# Result stored in integral_matrix variable
# Independent variable is r_matrix

r_matrix = np.linspace(1e-9, 2e-6, 100)
integral_matrix = []

# Matrix storing integration result: 
    
def tweezer(r_prime):
    # r' is r_prime
    integral = np.exp(-r_prime**2 / waist**2) * jv(0, wavenumber * r_prime * r_matrix[i] / eq_focal_length) * r_prime
    return integral

# Compute numerical integral as a function of r', save result by appending list

for i in range(len(r_matrix)):
    result , error = scipy.integrate.fixed_quad(tweezer, 0, waist)
    integral_matrix.append(result)
 
# Convert to np array, compute intensity, normalize

tweezer_array = np.array(integral_matrix)    
tweezer_intensity = abs(tweezer_array)**2
tweezer_intensity = tweezer_intensity / np.max(tweezer_intensity)

# Airy disk, point spread function (PSF)

airy = eq_focal_length / wavenumber / r_matrix * jv(1, wavenumber * r_matrix * aperture_radius / eq_focal_length)
airy_intensity = abs(airy)**2
airy_intensity = airy_intensity / np.max(airy_intensity)
    
#%% Plotting

fig, ax = plt.subplots(1, 1, figsize = (4, 3))
ax.grid()

# rescale x axis to show um instead of m
r_matrix = 1e6 * r_matrix

ax.plot(r_matrix, tweezer_intensity, label = 'point spread function')
ax.plot(r_matrix, airy_intensity, label = r'$w_i \sim R$')

ax.set_xlabel(r'radial coordinate in focal plane $r$ [$\mu$m]')
ax.set_ylabel('normalized intensity [a.u.]')
ax.legend()

plt.savefig('exports/GaussianVsPlanewaveAperture.pdf', dpi = 500)

