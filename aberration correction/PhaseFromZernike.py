#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 8 00:01:17 2021
@author: marijn

section of code copied from https://github.com/wavefrontshaping/WFS.net 
credits to the author
"""

#%% imports
import numpy as np
from aotools.functions import phaseFromZernikes
import matplotlib.pyplot as plt

#%%variables
resolution = [1920, 1200]
height = resolution[1]
width = resolution[0]

# zernike terms, various aberrations up to j = 14
zernike_terms = [0, # piston
                 0, # tilt 
                 0, # tip
                 0, # defocus
                 0, # oblique astig
                 1, # vert astig
                 0, # hor. coma
                 0, # vert. coma
                 0, # vert. trefoil
                 0, # oblique trefoil
                 0, # spherical
                 0, # second. astig
                 0, # oblique second. astig.
                 0, # vert. quadrafoil
                 0] # oblique quadrafoil

#%% zernike phase mask functions 
def get_disk_mask(shape, radius, center = None):
    '''
    Generate a binary mask with value 1 inside a disk, 0 elsewhere
    :param shape: list of integer, shape of the returned array
    :radius: integer, radius of the disk
    :center: list of integers, position of the center
    :return: numpy array, the resulting binary mask
    '''
    if not center:
        center = (shape[0] // 2, shape[1] // 2)
    X,Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    mask = (Y - center[0])**2 + (X - center[1])**2 < radius**2
    return mask.astype(int)
  
def complex_mask_from_zernike_coeff(shape, radius, center, vec):
    '''
    Generate a complex phase mask from a vector containting the coefficient of the first Zernike polynoms.
    :param DMD_resolution: list of integers, contains the resolution of the DMD, e.g. [1920,1200]
    :param: integer, radius of the illumination disk on the DMD
    :center: list of integers, contains the position of the center of the illumination disk
    :center: list of float, the coefficient of the first Zernike polynoms
    '''
    # Generate a complex phase mask from the coefficients
    zern_mask = np.exp(1j * phaseFromZernikes(vec, 2 * radius))
    # We want the amplitude to be 0 outside the disk, we fist generate a binary disk mask
    amp_mask = get_disk_mask([2 * radius] * 2, radius)
    # put the Zernik mask at the right position and multiply by the disk mask
    mask = np.zeros(shape = shape, dtype = np.complex64)
    mask[center[0] - radius:center[0] + radius,
         center[1] - radius:center[1] + radius] = zern_mask * amp_mask
    return mask

# Call functions to make mask with predefined zernike terms
complex_mask = complex_mask_from_zernike_coeff(shape = resolution,
                                               radius = resolution[1]//2,
                                               center = [width // 2, height // 2],
                                               vec = zernike_terms)

# make real valued phase mask, 8 bit depth
# real valued, transpose, round to integer
# add 128 to convert (-pi,pi) to (0,2pi)
real_mask = np.transpose(np.round(
        np.angle(complex_mask) * 2**8 / (2 * np.pi) + 2**7))
    
#%% Plotting
plt.imshow(real_mask)
