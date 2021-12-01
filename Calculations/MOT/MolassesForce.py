#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 23:37:28 2021

@author: marijn

Plots molasses force of a MOT
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

#%% variables. Dimensionless detuning delta/gamma

S = 2.5
detuning = - 1.5
plot_range = 4
data_points = 100

#%% Data

# F F/hky, 
# v=y/k here y is gamma

omega_d = np.linspace(-plot_range, plot_range, data_points)

def forces(detuning, omega_d):
    force_plus = 0.5 * S / (1 + S + 4 * (detuning - omega_d)**2)
    force_minus=-0.5 * S / (1 + S + 4 * (detuning + omega_d)**2)

    force_total = force_plus + force_minus
    
    return force_plus, force_minus, force_total

force_plus, force_minus, force_total = forces(detuning, omega_d)

#%% plotting

fix, ax = plt.subplots(figsize = (4, 3))
ax.grid(which='both')

ax.plot(omega_d, force_plus, 
        label = r'$\sigma_{+}$',
        linewidth = 2,
        linestyle = '--',
        color = 'blue')

ax.plot(omega_d, force_minus,
        label = r'$\sigma_{-}$',
        linewidth = 2,
        linestyle = '--',
        color = 'red')

ax.plot(omega_d, force_total,
        label = r'$\sigma_{+}+\sigma_{-}$',
        linewidth = 3,
        linestyle = '-',
        color = 'purple')

# Ticks, lines, etct. 

ax.set_xlabel(r'velocity [$\gamma/k$]', usetex = True, family = 'serif')
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.set_ylabel(r'force [$\hbar k \gamma$]', usetex = True, family = 'serif')

ax.legend()

ax.axhline(0, 
           color = 'black', 
           linewidth = .5)
ax.axvline(0, 
           color = 'black'
           , linewidth = .5)

#%% Saving

plt.savefig('exports/MOTplot.pdf', 
            dpi = 200, 
            pad_inches = 0,
            bbox_inches = 'tight')
