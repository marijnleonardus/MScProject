#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 23:39:23 2021

@author: marijn

script computes function of at focal plane for fixed w_i, R, lambda, f
"""

# Packages
import numpy as np
import scipy.integrate
from  scipy.special import jv
import matplotlib.pyplot as plt

# Constants, to be used in our objective
# We model the objective as a thin lens
w_i = 0.002
R = 0.002
lam = 780* 10**(-9)
k = 2*np.pi / lam
f = 0.004

# Radius in focal plane, as well as initializing empty tweezer matrix
r_matrix = np.linspace(1e-9, 1.5e-6,100)
tweezer_matrix = []

# Tweezer integral: r' is r_prime
def tweezer(r_prime):
    integral = np.exp(-r_prime**2 / w_i**2) * jv(0, k * r_prime * r_matrix[i] / f) * r_prime
    return integral

# Compute numerical integral as a function of r', save result by appending list
for i in range(len(r_matrix)):
    result , error = scipy.integrate.fixed_quad(tweezer, 0, w_i)
    tweezer_matrix.append(result)
 
# Convert to np array, compute intensity, normalize
tweezer_array = np.array(tweezer_matrix)    
tweezer_intensity = abs(tweezer_array)**2
tweezer_intensity_normalized = tweezer_intensity / np.max(tweezer_intensity)

# Airy 
airy = f / k / r_matrix * jv(1, k * r_matrix * R / f)
airy_intensity = abs(airy)**2
airy_intensity_normalized = airy_intensity / np.max(airy_intensity)
    
# Plot tweezer
fig, ax = plt.subplots(1,1, figsize = (6, 4))
ax.grid()

# rescale x axis to show um instead of m
r_matrix = 10**6 * r_matrix
ax.plot(r_matrix, tweezer_intensity_normalized, label = 'airy pattern')
ax.plot(r_matrix, airy_intensity_normalized, label = 'tweezer intensity')

ax.set_xlabel(r'radial coordinate in focal plane $r$ [$\mu$m]')
ax.set_ylabel('normalized intensity [a.u.]')
ax.legend()

# Saving
plt.savefig('tweezer_vs_airy.pdf', dpi = 500)
plt.show()
