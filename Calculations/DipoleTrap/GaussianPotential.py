#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Mar 19 14:55:26 2021

@author: Marijn Venderbosch

Script computes dipole potential from Gaussian beam
"""

#%% Imports

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as plticker
import numpy as np

#%% Variables

# dimensionless 
w_0 = 1
z_R = 1
factor = 1
power = 1

# Initialize r and z coordinates, radial and longitudal direction respectively
nz, nr = (1000, 1000)

plotrange_r = 2.5
plotrange_z = 2.5

spacing_r = 100
spacing_z = 100

#%% Create data

# Create 2D grid

def grid(plotrange_r, spacing_r, plotrange_z, spacing_z):
    rv = np.linspace(-plotrange_r, plotrange_r, spacing_r)
    zv = np.linspace(-plotrange_z, plotrange_z, spacing_z)

    r, z = np.meshgrid(rv,zv)
    
    return r,z

r, z = grid(plotrange_r, spacing_r, plotrange_z, spacing_z)

# Gaussian Beam

def gaussian_beam(w_0, z_R, r, z):
    
    # Compute beam width 
    w = w_0 * np.sqrt(1 + z**2 / z_R**2)
    
    # Compute intensity which is proportional to trap depth
    intensity = 2 * power * np.exp(-2 * r**2 / w**2) / (np.pi * w**2)
    
    return intensity

light_intensity = gaussian_beam(w_0, z_R, r, z)

# Trap depth is linear in light intensity, normalize

gaussian_potential = - factor * light_intensity
gaussian_potential =- gaussian_potential / np.min(gaussian_potential)
#%% Plotting

fig = plt.figure(figsize = (5.5, 3))
ax = fig.gca(projection = '3d')

surf = ax.plot_surface(r, z, gaussian_potential,
                       cmap = cm.viridis, 
                       linewidth = 0, 
                       antialiased = True,
                       vmin = -1,
                       vmax = 0
                       )

# Colorbar 

tickslist = np.linspace(gaussian_potential.min(),
                    gaussian_potential.max(),
                    3
                    )

cbar = fig.colorbar(surf, 
             shrink = 0.5,
             aspect = 9, 
             pad = 0.15,
             ticks = tickslist
             )

# Colorbar ticks

ax.set_xlabel(r'$r/w_0$')
ax.set_ylabel(r'$z/z_R$')
ax.set_zlabel(r'$U(r,z)/U_0$')
ax.grid()

# Ticks

loc_1 = plticker.MultipleLocator(base = 1) 
loc_0_25 = plticker.MultipleLocator(base = 0.25) 

ax.xaxis.set_major_locator(loc_1)
ax.yaxis.set_major_locator(loc_1)
ax.zaxis.set_major_locator(loc_0_25)


# Saving

plt.savefig('exports/GaussianTrapDepth.pdf',
            dpi = 300,
            bbox_inches = 'tight',
            pad_inches = 0
            )
