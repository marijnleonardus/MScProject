# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 17:11:48 2021

@author: Marijn Venderbosch

This script will fit a histogram from the trap depths and beamwiths as found from the script FitSpots.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# load fit paramters from FitSpots.py
sigmas = np.load('sigma.npy')
trap_depths = np.load('trap_depth.npy')

# compute beam widths
beam_width_pixels = 2 * sigmas
magnification = 60
# pixels are 4.65 micron. Magnification onto camera is 60X
# 2*sigma corresponds to the 1/e^2 radius
beam_width_microns = beam_width_pixels * 4.65 / 60

mu_beam_width, stddev_beam_width = norm.fit(beam_width_microns)
mu_trap_depth, stddev_trap_depth = norm.fit(trap_depths)

# Histogram plots 
n_bins = 5
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout = True)
n_beamwidth, bins_beamwidth, patches_beamwidth = ax1.hist(beam_width_microns, bins = n_bins)
n_trapdepth, bins_trapdepth, patches_trapdepth = ax2.hist(trap_depths, bins = n_bins)

xmin_beamwidth, xmax_beamwidth = ax1.get_xlim()
xmin_trapdepth, xmax_trapdepth = ax2.get_xlim()

x_beamwidth = np.linspace(xmin_beamwidth, xmax_beamwidth, 100)
x_trapdepth = np.linspace(xmin_trapdepth, xmax_trapdepth, 100)

gauss_beamwidth = norm.pdf(bins_beamwidth, mu_beam_width, stddev_beam_width)
gauss_trapdepth = norm.pdf(bins_trapdepth, mu_trap_depth, stddev_trap_depth)

ax1.plot(x_beamwidth, gauss_beamwidth, 'k', linewidth = 2)
ax2.plot(x_trapdepth, gauss_trapdepth, 'k', linewidth = 2)

plt.show()
