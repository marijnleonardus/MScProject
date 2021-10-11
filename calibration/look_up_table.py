# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:08:08 2021

@author: Marijn Venderbosch

This script plots the calibrated .lut values. Optionally, can also plot factory provided .lut slm5952......
"""

#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% read data

# Factory supplied .lut, commented for now
# df_813 = pd.read_csv('lut files/slm5953_at785_HDMI.lut', delimiter = ' ', header = None)
# lut_813_mV = np.array(df_813)
# lut_820 = lut_813_mV * 5 / 2**(12)

# Own diffractive lut calibration file
df_820 = pd.read_csv('lut files/lut820_diffractivecalibration.lut', delimiter = ' ', header = None)
lut_820_mV = np.array(df_820)

# Convert bit values to V values. 0-5V in 4096 (2^12) steps
lut_820 = lut_820_mV * 5 / 2**(12)

#%%plotting
#plt.plot(lut_813[:, 1], label = '813.lut')
fig, ax = plt.subplots(figsize = (4,3))
ax.plot(lut_820[:, 1], label = '820.lut')

ax.grid()
ax.set_xlabel('grey level 0-255')
ax.set_ylabel('voltage response [a.u]')
#plt.legend()

#%% saving
plt.savefig('lut_plot.pdf', dpi = 300, bbox_inches = 'tight')
