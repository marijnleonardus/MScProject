#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 22:48:06 2021

@author: marijn

As a result of beam waist (w ~ R), longitudinal profile will not be
diffraction-limited PSF

We calculate this deviation and compare to ideal PSF

"""

#%%imports

import numpy as np
import matplotlib.pyplot as plt

#%% variables

lam = 820e-9
k = 2 * np.pi / lam
f = 4e-3
R = 2e-3
w_i = 2e-3
plot_range = 15e-6

#%% Data and functions

# Mathemetica result and PSF
# in terms of dimenionless defocus paramter u = k dz R**2/f**2

dz = np.linspace(-plot_range, plot_range, 1000)
z_data = k * dz * R**2 / f**2

def longitudinal_intensity(u):
    # Formula from DOI: 10.1088/978-1-6817-4337-0ch2
    intensity = (-2 * np.exp(1) * np.cos(u / 2) + np.exp(2) + 1) / (np.exp(2)* (u**2 + 4))
    # Normalize
    intensity_normalized = intensity / np.max(intensity)
    return intensity_normalized

z_intensity = longitudinal_intensity(z_data)

def ideal_PSF(u):
    PSF = np.abs(np.sin(u / 4)/(u/4))**2
    # Normalize
    PSF_normalized = PSF / np.max(PSF)
    return PSF_normalized

PSF_intensity = ideal_PSF(z_data)

#%% Plotting

# 1D plot of theory vs measurement
fig, ax = plt.subplots(figsize = (4, 2.5))
plt.grid()

# Rescale x axis to microns
dz_microns = dz * 10e5

plt.plot(dz_microns, z_intensity,
         label = r'$w_i \sim R$'
         )

plt.plot(dz_microns, PSF_intensity, 
         label = 'Plane wave')

ax.set_xlabel(r'$\delta z$ [$\mu$m]', usetex = True)
ax.set_ylabel(r'$I/I_0$', usetex = True)
ax.legend()




#%% Saving
plt.savefig('exports/LongitudinalTweezerField.pdf',
            dpi = 200, 
            pad_inches = 0, 
            bbox_inches = 'tight'
            )
