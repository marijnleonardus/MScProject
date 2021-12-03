# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:34:47 2021

@author: Marijn Venderbosch

Correction on Gaussian fit of PSF in non-paraxial case

Formula from Zhang, 2007: https://doi.org/10.1364/AO.46.001819
"""

import numpy as np

# Variables

n = 1
alpha = np.arcsin(0.5) / n

#%% Radial correction

# Computes sigma = factor * wavelength. Make dimensionless by dividing by wavelength
wavelength = 1
k = 2 * np.pi / wavelength

# Formula from Zhang2007 paper
# Dimensionless because of deviding by wavelength
# So it computes a factor according to sigma = factor * wavelength

def non_paraxial_sigma_radial(a, n, k):
    
    # n refractive index
    # k wavenumber
    # a maximum half angle of light cone
    
    fraction = (4 - 7 * np.cos(a)**(3/2) + 3 * np.cos(a)**(7/2)) / (7 * (1 -  np.cos(a)**(3/2)))
    sigma = fraction**(-0.5) / (n * k)
    return sigma

factor_radial = non_paraxial_sigma_radial(alpha, n, k)
print(factor_radial)

#%% Axial correction

# uses eq. from table 3 of same paper Zhang 2007

def non_paraxial_sigma_axial(a, n, k):
    cos = np.cos(a)
    
    numerator = 5 * np.sqrt(7) * (1 - cos**1.5)
    denominator = np.sqrt(6) * n * k * (4 * cos**5 - 25 * cos**3.5 + 42 * cos**2.5 - 25 * cos**1.5 + 4)**.5
    
    sigma = numerator / denominator           
    return sigma

factor_axial = non_paraxial_sigma_axial(alpha, n, k)

# We want to convert to the Rayleigh range

sigma_axial = factor_axial * 820
rayleigh = sigma_axial * np.sqrt(2 * np.log(2))






             