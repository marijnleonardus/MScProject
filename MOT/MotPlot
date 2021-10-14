#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 23:37:28 2021

@author: marijn

Plots field of a MOT
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

#%% variables. Dimensionless detuning delta/gamma
S = 2
detuning = - 1
plot_range = 4

#%% dimensionless force F/hky, v=y/k here y is gamma
omega_d = np.linspace(-plot_range, plot_range, 100)

force_plus = 0.5 * S / (1 + S + 4 * (detuning - omega_d)**2)
force_minus=-0.5 * S / (1 + S + 4 * (detuning + omega_d)**2)
force_total = force_plus + force_minus

#%% plotting
fix, ax = plt.subplots(figsize = (4, 3))


ax.plot(omega_d, force_plus, 
        label = r'$F^+$',
        linewidth = 2,
        linestyle = '--',
        color = 'blue')
ax.plot(omega_d, force_minus,
        label = r'$F^-$',
        linewidth = 2,
        linestyle = '--',
        color = 'red')
ax.plot(omega_d, force_total,
        label = r'$F^++F^-$',
        linewidth = 3,
        linestyle = '-',
        color = 'purple')

ax.set_xlabel(r'$vk/ \gamma$')
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.set_ylabel(r'$F / \hbar k \gamma$')

ax.legend()
ax.grid(which='both')

plt.savefig('MOT_plot.pdf', 
            dpi = 300, 
            bbox_inches = 'tight')
