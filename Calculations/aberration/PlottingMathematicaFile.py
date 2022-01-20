#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 21:12:56 2022

@author: marijn

This script plots zernike polynomials from Mathematica script in same folder

In the mathematica file, the trig relation is Taylor expanded in 2nd and 4th
order terms in terms of the angle
"""

# %%imports

import numpy as np
import matplotlib.pyplot as plt

#%% variables

wavelength = 820e-9
k = 2 * np.pi / wavelength
n0 = 1
NA = 0.5

# Mitutoyo corrected for plate of 
ncorr = 1.5105
dcorr = 3.5e-3

# our glass plate
nquartz = 1.453
dplate = 4e-3

#%% calculations

# angle variable
t = np.linspace(0, np.arcsin(NA), 101)
rho = np.linspace(0, 1, 101)

# path length difference because of 4 mm glass 1.453, not corrected
wavesGlass = 450.859 * t**2 + 122.595 * t**4

# path length difference after correction mitutoyo subtracted
wavesCorrected = 45.9178 * t**2 + 23.1342 * t**4

Z04 = 23.1342 * (np.arcsin(NA * 1))**4
Z04polynomial = Z04 / 6 * (1 - 6 * rho**2 + 6 * rho**4)


#%% plotting, saving

fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, 
                               nrows = 1,
                               tight_layout = True,
                               figsize = (8, 3))

ax1.plot(t, wavesGlass, label = r'full aberration')
ax2.plot(t, wavesCorrected, label = r'correction error')
ax3.plot(rho, Z04polynomial, label = r'only $R_4^0$')



# shared y axis labels
ax1.set_ylabel(r'$\phi(\alpha_0)/2\pi$', usetex = True)

# x labels
ax1.set_xlabel(r'$\alpha_0$', usetex = True)
ax2.set_xlabel(r'$\alpha_0$', usetex = True)
ax3.set_xlabel(r'$r/R$', usetex = True)

# legend
ax1.legend(loc = 'upper center')
ax2.legend(loc = 'upper center')
ax3.legend(loc = 'upper center')

# annotate
ax1.annotate("(a)", xy = (0.455, -0.3), xycoords = "axes fraction", fontweight = 'bold', fontsize = 9)
ax2.annotate("(b)", xy = (0.455, -0.3), xycoords = "axes fraction", fontweight = 'bold', fontsize = 9)
ax3.annotate("(c)", xy = (0.455, -0.3), xycoords = "axes fraction", fontweight = 'bold', fontsize = 9)

plt.savefig('exports/SphericalAberrationTerms.pdf',
            pad_inches = 0,
            bbox_inches = 'tight',
            dpi = 300)
