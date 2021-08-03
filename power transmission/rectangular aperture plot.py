# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:39:44 2021

@author: Marijn Venderbosch

power tranmission rectangular aperture
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special 

# Variables normalized in terms of beam width
beam_width = 1
S_x = beam_width
# SLM has dimensions 8 x 15 mm (vertically orientated)
proportion = 15 / 8
S_y = proportion * S_x

# Defining independent variable beam width
start = 0.001
stop = 3
steps = 100
width = np.linspace(start, stop, steps)

# Calculate power transmission, which is integration over I(x,) in one dimension
# Derived by hand and checked with mathematica, this is only the final result
def transmission(w, Sx):
        trans = scipy.special.erf(np.sqrt(2) * Sx / w)
        return trans

# Calculate x,y power transmission
x_transmission = transmission(width, S_x)  
y_transmission = transmission(width, (proportion * S_x))

# Total transmission is the product of x and y 
total_transmission = x_transmission * y_transmission

# Initialize plot
fig, ax = plt.subplots(1, 1, figsize = (7,5))
ax.grid()

# Plot x, y and total transmissions
ax.plot(width, x_transmission, '--', label = r'$x$-direction')
ax.plot(width, y_transmission, '--', label = r'$y$-direction')
ax.plot(width, total_transmission, '-', label = 'total')

# Plot line at x = S_x
ax.axvline(x = 1, color = 'grey', linestyle = 'solid')

# Set lables, title
ax.set_xlabel(r'$w/S_x$')
ax.set_ylabel(r'$P/P_0$ [%]')
ax.set_title('Power transmission'
             '\n'
             r'aperture size $(2S_x,2S_y) =(8,15)$ mm')
ax.legend()

# Save plot
plt.savefig('transmission_rectangular.pdf', tight_layout = True)

# Zoomed in plot
fig, ax = plt.subplots(1, 1, figsize = (7,5))
ax.grid()

# Plot x, y and total transmissions
ax.plot(width, total_transmission, '-', label = r'transmission in $x$ and $y$')

# Plot line at x = S_x
ax.axvline(x = 1, color = 'grey', linestyle = 'solid')

# Set lables, title
ax.set_xlabel(r'$w/S_x$')
ax.set_ylabel(r'$P/P_0$ [%]')
ax.set_title('Power transmission'
             '\n'
             r'aperture size $(2S_x,2S_y) =(15,8)$ mm')
ax.set_xlim(0.75, 1.25)
ax.set_ylim(0.8, 1.05)
ax.legend()

# Save and show
plt.savefig('transmission_rectangular_zoomed.pdf', tight_layout = True)
plt.show()

