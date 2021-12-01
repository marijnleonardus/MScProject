# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:39:44 2021

@author: Marijn Venderbosch

power tranmission rectangular aperture
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.special 

#%% variables

# SLM has dimensions 8 x 15 mm (vertically orientated)
# Dimensions normalized by short-axis S_x

S_x = 1 # short semi-axis
proportion = 15.36 / 9.6
S_y = S_x * proportion

# Defining independent variable beam width, ranging from 

def beam_width_list():
    start = 0.1 
    stop = 3
    steps = 100
    normalized_beam_width = np.linspace(start, stop, steps)
    return normalized_beam_width

waist = beam_width_list()

#%% calculations

# Calculate power transmission, which is integration over I(x) in one dimension
# Derived by hand and checked with mathematica, this is only the final result

def transmission(beam_width, short_axis):
        transmission = scipy.special.erf(np.sqrt(2) * short_axis / beam_width)
        return transmission

def total_transmission(beam_width, short_axis, long_axis):
    
    # Calculate x,y power transmission
    transmission_horizontal = transmission(waist, S_x)
    transmission_vertical = transmission(waist, S_y)
    
    # Total transmission is the product of x and y 
    total_transmission = transmission_horizontal * transmission_vertical
    return total_transmission

total_transmission = total_transmission(waist, S_x, S_y)

#%% plotting

# Initialize plot

fig1, ax1 = plt.subplots(1, 1, figsize = (4, 3))
ax1.grid()

# Plot x, y and total transmissions
ax1.plot(waist, total_transmission, '-', label = r'$P/P_0$')

# Plot line at x = S_x
ax1.axvline(x = 1, color = 'grey', linestyle = 'solid')

# Set lables, title

ax1.set_xlabel(r'$w/S_x$')
ax1.set_ylabel(r'$P/P_0$ [%]')
ax1.legend()

# Save plot and show
plt.savefig('transmission_rectangular.pdf',
            bbox_inches = 'tight',
            dpi = 300)
plt.show()
