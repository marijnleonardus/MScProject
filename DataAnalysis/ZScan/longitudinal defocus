#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 22:48:06 2021
@author: marijn
"""

#%%imports
import numpy as np
import matplotlib.pyplot as plt

# variables
lam = 780e-9
k = 2 * np.pi / lam
f = 4e-3
R = 2e-3
w_i = 2e-3

#%% Mathemetica result and PSF
# in terms of dimenionless defocus paramter u = k dz R**2/f**2

plot_range = 15e-6
dz = np.linspace(-plot_range, plot_range, 1000)
u = k * dz * R**2 / f**2

intensity = (-2 * np.exp(1) * np.cos(u / 2) + np.exp(2) + 1) / (np.exp(2)* (u**2 + 4))
intensity_normalized = intensity / np.max(intensity)

# PSF
PSF = np.abs(
    np.sin(u / 4)/(u/4)
    )**2
PSF_normalized = PSF / np.max(PSF)

#%%ploting
fig, ax = plt.subplots(figsize = (4,3))

# Rescale x axis to microns
dz_microns = dz * 10e5

plt.plot(dz_microns, intensity_normalized, label = 'Tweezer')
plt.plot(dz_microns, PSF_normalized, label = 'PSF')

plt.grid()
ax.set_xlabel(r'defocus [$\mu$m]')
ax.set_ylabel(r'Normalized intensity [a.u.]')
ax.legend()

plt.savefig('Longitudinal_defocus_PSF.pdf')
