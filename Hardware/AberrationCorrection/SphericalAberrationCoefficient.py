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
coefficient = 0.2535 # zernike coefficient as applied onto the SLM

#%% calculation

# independent varible rho
# dimenionless radial coordinate r/R
# Same as SLM plane as objective back aperture plane (conjugated)

rho = np.linspace(0, 1, 101)

# Eq. (7) from Iwaniuk 2007
# Vol. 19, No. 20 / OPTICS EXPRESS 19407

# Taking both ^2 and ^4 term of the formula in the paper
correctionPaper = np.pi * d * (1 - nr) / wavelength * (-1 * rho**2 * NA**2 * (1 - nr**2)
                                           + 3 * rho**4 * NA**4 / 4 * (1 - nr**2)**2)

# Only taking ^4 term
Z44 = np.pi * d * (1 - nr) / wavelength * (3/4* rho**4 * NA**4 * (1 - nr**2)**2)

# Expanding in Z00, Z22 and Z40
wavefront = coefficient - 12.93 * rho**2 + coefficient * rho**4
Z04 = coefficient * (1 - 6 * rho**2 + 6 * rho**4)

# Divide by 2 pi to get amount of phase shifts
wavesCorrectionPaper = correctionPaper / (2 * np.pi)
wavesZ04 = Z04 / (2 * np.pi)
wavesZ44 = Z44 / (2 * np.pi)

#%% plotting, saving

fig, (axPaper, axZ04, axZ44) = plt.subplots(ncols = 3, 
                               nrows = 1,
                               tight_layout = True,
                               figsize = (8, 3))

axPaper.plot(rho, wavesCorrectionPaper, label = r'$\phi(\rho)$')
axZ04.plot(rho, wavesZ04, label = r'$Z_0^4$ term')
axZ44.plot(rho, wavesZ44, label = r'$\rho^4$ term')

# shared y axis labels
axPaper.set_ylabel(r'$\phi(\rho)/2\pi$', usetex = True)

# x labels
axPaper.set_xlabel(r'$r/R$', usetex = True)
axZ04.set_xlabel(r'$r/R$', usetex = True)
axZ44.set_xlabel(r'$r/R$', usetex = True)

# legend
axPaper.legend(loc = 'upper right')
axZ04.legend(loc = 'upper center')
axZ44.legend(loc = 'upper left')

# annotate
axPaper.annotate("a)", xy = (0.1, 0.1), xycoords = "axes fraction", fontweight = 'bold', fontsize = 12)
axZ04.annotate("b)", xy=(0.1, 0.1), xycoords = "axes fraction", fontweight = 'bold', fontsize = 12)
axZ44.annotate("c)", xy=(0.1, 0.1), xycoords = "axes fraction", fontweight = 'bold', fontsize = 12)


plt.savefig('exports/SphericalAberrationTerms.pdf',
            pad_inches = 0,
            bbox_inches = 'tight',
            dpi = 300)
