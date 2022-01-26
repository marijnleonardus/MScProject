#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 13:53:23 2022

Plots sequence timings using broken axis

@author: marijn
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

#%% variables

# time
t = np.linspace(0, 550, 701)

motLoad = 400
Field = 1500
dipoleLoad = 150
cameraDelay = 10
exposure = 100

# binary value (plot amplitude)
value = 1.2

# plotting
xOffset = -300
yOffset = 0.3

#%% data

# components of the sequence to plot
motBeams = value-np.heaviside(t - motLoad, 1) * value
TitaniumSapphire = np.heaviside(t - motLoad + dipoleLoad, 1) * value
camera = np.heaviside(t - motLoad - cameraDelay, 1) * value - np.heaviside(t - motLoad - cameraDelay - exposure, 1) * value

#%% plotting and saving

# two seperate axis that plot differnent data, but one is zoomed in on different segment
# this is 'axzoom' 

fig, ax = plt.subplots(ncols = 1,
                       nrows = 1,
                       figsize = (4, 2),
                       sharey = True)

ax.set_ylim(-7, 2)
# grey vertical line
ax.axvline(motLoad,
           color = 'grey',
           alpha = 0.6,
           linestyle = '--')

# plot step function and annotation
def plotWithText(time, component, height, text, xOffset):
    ax.plot(time, component + height,
            linewidth = 2.5)
    ax.text(xOffset, height + yOffset, text)
    
plotWithText(t, motBeams, 0, 'MOT beams', xOffset)
plotWithText(t, TitaniumSapphire, -2, 'optical dipole trap', xOffset)
plotWithText(t, camera, -4, 'probe beam', xOffset)
plotWithText(t, camera, -6, 'sCMOS shutter', xOffset)

# title, ticks
ax.set_xlabel('time [ms]')
ax.yaxis.set_ticklabels([])

# saving
plt.savefig('exports/Sequence.pdf',
            dpi = 300, 
            pad_inches = 0, 
            bbox_inches = 'tight')
