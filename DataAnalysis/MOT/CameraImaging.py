# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 13:38:29 2021

@author: Marijn Venderbosch

calculate relevant parameters for installing second MOT camera 

Desired specs:
    total length < 40 cm
    magnification ~ 1-2
    d_1 > 11 cm
"""

#%% Imports

import numpy as np

#%% Variables

f = 5 # focal length [cm]
d_1 = 11 # distance tweezer to first lens [cm]
n = 1.0 # refractive index air
R = 2.5 / 2 # radius lens [cm]
wavelength = 820e-9 # TiS laser wavelength [nm]

#%% Parameters to be calculated

def lens_formula(f, d_1):
    # d_2 is distance lens to camera
    d2 = (1 / f - 1 / d_1)**(-1)
    return d2

def numerical_aperture(R, d_1):
    theta = np.arctan(R / d_1)
    NA = n * np.sin(theta)
    return NA

# Magnification
magnification = lens_formula(f, d_1) / d_1

# Spot size TiS laser
diffraction_limit = wavelength * .61 / numerical_aperture(R, d_1)

# Total length system
total_length = d_1 + lens_formula(f, d_1)

#%% Print result

print("total length: " + str(total_length))
print("magnification: " + str(magnification))
print("diffraction limit is: " + str(diffraction_limit))
