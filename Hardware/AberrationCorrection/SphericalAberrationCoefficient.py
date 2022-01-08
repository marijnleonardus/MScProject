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
deff = d / nquartz

#%% calculation

# independent varible rho
# dimenionless radial coordinate r/R
# Same as SLM plane as objective back aperture plane (conjugated)

rho = np.linspace(0, 1, 101)

# Eq. (7) from Iwaniuk 2007
# Vol. 19, No. 20 / OPTICS EXPRESS 19407

phi = np.pi * deff * (1 - nr) / wavelength * (3 * rho**4 * NA**4 / 4 * (1 - nr**2)**2)

# Divide by 2 pi to get amount of phase shifts
waves = phi / (2 * np.pi)

#%% plotting, saving

fig, ax = plt.subplots(figsize = (3, 2))
ax.plot(rho, waves)

ax.set_xlabel(r'$r/R$', usetex = True)
ax.set_ylabel(r'$\phi(r)/2\pi$', usetex = True)

plt.savefig('exports/SphericalAberrationTerm.pdf',
            pad_inches = 0,
            bbox_inches = 'tight',
            dpi = 300)
