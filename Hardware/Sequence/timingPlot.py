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
t = np.linspace(0, 2000, 2001)

motLoad = 1500
Field = 1500
dipoleLoad = 150
cameraDelay = 10
exposure = 100

# binary value (plot amplitude)
value = 1.2

# plotting
xOffset = 100
yOffset = 0.3

#%% data

# components of the sequence to plot
motBeams = 1-np.heaviside(t - motLoad, 1) * value
TitaniumSapphire = np.heaviside(t - motLoad + dipoleLoad, 1) * value
camera = np.heaviside(t - motLoad - cameraDelay, 1) * value - np.heaviside(t - motLoad - cameraDelay - exposure, 1) * value

#%% plotting and saving

# two seperate axis that plot differnent data, but one is zoomed in on different segment
# this is 'axzoom' 

fig, (ax, axzoom) = plt.subplots(ncols = 2,
                       nrows = 1,
                       figsize = (5, 2),
                       sharey = True)

# horizontal space
fig.tight_layout(pad = -0.5)

# domain of the two segments
ax.set_xlim(0, 1300)
axzoom.set_xlim(1300, 1650)

ax.set_ylim(-7, 2)
# grey vertical line
axzoom.axvline(motLoad,
           color = 'grey',
           alpha = 0.6,
           linestyle = '--')

# plot step function and annotation
def plotWithText(time, component, height, text, xOffset):
    ax.plot(time, component + height,
            linewidth = 2.5)
    axzoom.plot(time, component + height, 
                linewidth = 2.5)
    ax.text(xOffset, height + yOffset, text)
    
plotWithText(t, motBeams, 0, 'MOT beams', xOffset)
plotWithText(t, TitaniumSapphire, -2, 'optical dipole trap', xOffset)
plotWithText(t, camera, -4, 'probe beam', xOffset)
plotWithText(t, camera, -6, 'sCMOS shutter', xOffset)

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
axzoom.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright = 'off')
axzoom.yaxis.tick_right()

# plot breaking lines
d = .015 # how big to make the diagonal lines in axes coordinates

# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform = ax.transAxes, color = 'k', clip_on = False)
ax.plot((1-d,1+d), (-d,+d), **kwargs)
ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

kwargs.update(transform = axzoom.transAxes)  # switch to the bottom axes
axzoom.plot((-d,+d), (1-d,1+d), **kwargs)
axzoom.plot((-d,+d), (-d,+d), **kwargs)

# title, ticks
fig.suptitle('time [ms]', 
          y = -0.08,
          fontsize = 10)
axzoom.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_ticklabels([])

# saving
plt.savefig('exports/Sequence.pdf',
            dpi = 300, 
            pad_inches = 0, 
            bbox_inches = 'tight')
