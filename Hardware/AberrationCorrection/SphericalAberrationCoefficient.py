#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 21:12:56 2022

@author: marijn

Script computes Zernike coefficient to be used by SLM to correct spherical
aberration induced by incorrect cover glass thhickness
"""

# %%imports

import numpy as np
import matplotlib.pyplot as plt

#%% variables

wavelength = 820e-9
n0 = 1.003
n = 1.51
nquartz = 1.44
nr = n0 / n
NA = 0.5
d = 0.5e-3

#%% calculation

# independent varible rho
# dimenionless radial coordinate r/R
# Same as SLM plane as objective back aperture plane (conjugated)

rho = np.linspace(0, 1, 101)

# Eq. (7) from Iwaniuk 2007
# Vol. 19, No. 20 / OPTICS EXPRESS 19407

# Taking both ^2 and ^4 term
Z04 = np.pi * d * (1 - nr) / wavelength * (-1 * rho**2 * NA**2 * (1 - nr**2) + 3 * rho**4 * NA**4 / 4 * (1 - nr**2)**2)

# Only taking ^4 term
Z44 = np.pi * d * (1 - nr) / wavelength * (3/4* rho**4 * NA**4 * (1 - nr**2)**2)


# Divide by 2 pi to get amount of phase shifts
wavesZ04 = Z04 / (2 * np.pi)
wavesZ44 = Z44 / (2 * np.pi)

#%% plotting, saving

fig, (ax1, ax2) = plt.subplots(ncols = 2, 
                               nrows = 1,
                               tight_layout = True,
                               figsize = (5, 2.5))
ax1.plot(rho, wavesZ04)
ax2.plot(rho, wavesZ44)

ax1.set_xlabel(r'$r/R$', usetex = True)
ax2.set_ylabel(r'$\phi(\rho)/2\pi$', usetex = True)

ax2.set_xlabel(r'$r/R$', usetex = True)
ax1.set_ylabel(r'$\phi(\rho)/2\pi$', usetex = True)



plt.savefig('exports/SphericalAberrationTerm.pdf',
            pad_inches = 0,
            bbox_inches = 'tight',
            dpi = 300)
