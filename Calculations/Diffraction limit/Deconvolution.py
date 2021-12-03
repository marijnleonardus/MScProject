# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:34:47 2021

@author: Marijn Venderbosch

Computes size of spot corrected for deconvolution with imaging objective

Uses formula from Zhang2007 to estimate diffraction limit of 0.85 objective
used for imaging
"""

import numpy as np

#%% Variables

wavelength = 820 # nm
wavenumber = 2 * np.pi / wavelength
numerical_aperture_newport = 0.85
n = 1
alpha_newport = np.arcsin(numerical_aperture_newport)

#%% Calculations

# Formula from Zhang2007 paper

def sigma(a, n, k):
    fraction = (4 - 7 * np.cos(a)**(3/2) + 3 * np.cos(a)**(7/2)) / (7 * (1 -  np.cos(a)**(3/2)))
    sigma = fraction**(-0.5) / (n * k)
    return sigma

sigma_sys = sigma(alpha_newport, n, wavenumber)    

# Find sigma_image from spot fitting script 

sigma_image = 800 / 2

# Calculate object sigma and waist

sigma_object = np.sqrt(sigma_image**2 - sigma_sys**2)
waist_object = 2 * sigma_object
print("waist is: " + str(waist_object))