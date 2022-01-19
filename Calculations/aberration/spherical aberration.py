#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 23:09:20 2022

Script computes focal shift in waves as a result of a different
 - glass thickness, d
 - refractive index, n
 
than the objective was corrected for

@author: Marijn
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt

#%% Variables

NA = 0.5
wavelength = 820e-9

# Our glass cell
nquartz = 1.44
d = 3.9e-3

# What mitutoyo is corrected for
ncorr = 1.51
dcorr = 3.5e-3

#%% functions
rho = np.linspace(0, 1, 100)

def fourthorder(n, d, rho):
    return np.pi * d * (1 - 1 / n) / wavelength * (3 / 4* rho**4 * NA**4 * (1 - 1 / n**2)**2)


dphi = fourthorder(nquartz, d, rho) - fourthorder(ncorr, dcorr, rho)
waves = -dphi / (2 * np.pi)

#%% plotting

plt.plot(rho,waves)