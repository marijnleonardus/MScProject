# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:16:43 2021

@author: Marijn Venderbosch

circular aperture
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.special 

# Variables normalized in terms of beam width
beam_width = 1
R = beam_width

# Defining independent variable beam width
start = 0.001
stop = 2
steps = 100
width = np.linspace(start, stop, steps)

# Calculate power transmission, which is integration over I(r) * dA
# Standard result power trough aperture
def transmission(w, R):
        trans = 1 - np.exp(-2 * R**2 / w**2)
        return trans

# Initialize plot
fig, ax = plt.subplots(1, 1, figsize = (7,5))
ax.grid()

# Plot x, y and total transmissions
ax.plot(width, transmission(width, R), '-', label = r'transmission')

# Plot line at x = S_x
ax.axvline(x = 1, color = 'grey', linestyle = 'solid')

# Set lables, title
ax.set_xlabel(r'$w/R$')
ax.set_ylabel(r'$P/P_0$ [%]')
ax.set_title('Power transmission'
             '\n'
             r'Aperture radius $R = 2$ mm')
ax.legend()
ax.set_xlim(0.25, stop)

# Save and show
plt.savefig('transmission_circular.pdf', tight_layout = True)
plt.show()

