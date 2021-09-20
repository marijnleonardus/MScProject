## -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:55:26 2021
@author: Marijn Venderbosch
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Initialize r and z coordinates, radial and longitudal direction respectively
nz, nr = (100, 100)

# Amount of waists/rayleigh Ranges to include
plotrange_r = 3
plotrange_z = 3

# Amount of points for 2D grid
spacing_r = 100
spacing_z = 100

# Create 2D grid
rv = np.linspace(-plotrange_r, plotrange_r, spacing_r)
zv = np.linspace(-plotrange_z, plotrange_z, spacing_z)
r, z = np.meshgrid(rv,zv)

# Define optical potential function. Modeled as gaussian with rayleight range z_R
# and waist w_0. 
def optical_potential(U0, w0, zR, z, r):
    prefactor = -U0 / (1 + z**2/zR**2)
    exponent = -2*r**2 / (w0**2*(1 + z**2 / zR**2))
    return prefactor * np.exp(exponent)

# Return dimensionless optical potential. Dimensionless because w_i = z_R = U_0 = 1
potential = optical_potential(1, 1, 1, z, r)

# Plotting
fig = plt.figure(figsize = (5,4))
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(r ,z, potential,
                       cmap = cm.viridis, 
                       linewidth = 0, 
                       antialiased = False)
ax.set_zlim(-1.1,0)
cb = plt.colorbar(surf, 
                  ax = [ax], 
                  shrink = 0.45, 
                  aspect = 6, 
                  location ='left',
                  ticks=[-1,-0.8,-0.6,-0.4,-0.2,0],
                  pad = 0)

ax.set_xlabel(r'$r/w_0$')
ax.set_ylabel(r'$z/z_R$')
ax.set_zlabel(r'$U(r,z)/U_0$')
ax.grid()

# Saving
plt.savefig('ODTdepth.png', 
            bbox_inches='tight',
            pad_inches = 0,
            dpi = 500)
plt.show()
