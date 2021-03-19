# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:55:26 2021

@author: Marijn Venderbosch
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Problem is dimensionless
w_0 = 1
z_R = 1

# prefactor for later
factor = 1
power = 1

# Initialize r and z coordinates, radial and longitudal direction respectively
nz, nr = (100, 100)

plotrange_r = 3
plotrange_z = 3

spacing_r = 100
spacing_z = 100

# Create 2D grid
rv = np.linspace(-plotrange_r, plotrange_r, spacing_r)
zv = np.linspace(-plotrange_z, plotrange_z, spacing_z)

r, z = np.meshgrid(rv,zv)

# Compute beam width and intensity
w = w_0 * np.sqrt(1 + z**2/z_R**2)
intensity = 2 * power * np.exp(-2 * r**2 / w**2) / (np.pi * w**2)

# Trap depth
U = -factor * intensity

# Plotting
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(r ,z, U, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_zlabel(r'Trap depth $U(r,z)$')
ax.grid()

# Saving
plt.savefig('ODTdepth.png', dpi = 1000)
plt.show()