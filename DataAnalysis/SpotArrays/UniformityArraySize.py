# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:20:07 2022

@author: Marijn Venderbosch

computes array unifomrity as a function of amount of spots

uniformity computed by FindingAndFittingSpots.py
"""

#%% Imports

import matplotlib.pyplot as plt
import numpy as np

# Data from other script

spotAmount = np.linspace(2, 9, 8)
uniformity = [0.9, 9.5, 7.2, 7.8, 10.9, 10, 10.1, 12.7]

# Change to falsely reject H0 hypothesis that data is normally distrubited
# Not plotted
ShapiroWilk = [1, 0.789, 0.36, 0.96, 0.32, 0.9, 0.3, 0.65]


# Plot

fig, ax = plt.subplots(1, 1, figsize = (4.5, 2.5))
ax.grid()

ax.plot(spotAmount, uniformity,
        linewidth = 2.5)

ax.set_ylim(0, 13)
ax.set_xlabel(r'$n$', usetex = True)
ax.set_ylabel(r'$\sigma_{U_0}/\left\langle U_0 \right\rangle [\%]$', usetex = True)

plt.savefig('exports/UniformityAmountsSpots.pdf',
            dpi = 300, 
            pad_inches = 0,
            bbox_inches = 'tight')