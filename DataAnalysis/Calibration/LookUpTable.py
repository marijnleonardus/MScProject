# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:08:08 2021

@author: Marijn Venderbosch

This script plots the calibrated .lut values. 
Optionally, can also plot factory provided .lut slm5952......

Also, plots the raw data the lut is based on 
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
df_820 = pd.read_csv('files/lut820_diffractivecalibration.lut',
                     delimiter = ' ',
                     header = None)
lut_820_mV = np.array(df_820)
# Convert bit values to V values. 0-5V in 4096 (2^12) steps
lut_820 = lut_820_mV[:,1] * 5 / 2**(12)

# Raw data of calibration, power in first order measurement
df_raw_data = pd.read_csv('files/Raw0.csv',
                          delimiter = ',',
                          header = None)
raw_array = np.array(df_raw_data)
raw_normalized = np.array(raw_array[:, 1]) / np.max(raw_array[:, 1])

#%%plotting
#ax.plot(lut_813[:, 1], label = '813.lut')
fig, ax = plt.subplots(figsize = (3.3, 2.2))
line1 = ax.plot(lut_820,
        label = 'LUT',
        color = 'navy')

ax.grid()
ax.set_xlabel('grey level 0-255')
ax.set_ylabel('voltage response [V]',
              color = 'navy')
ax.set_ylim(-0.2, 5.2)
ax.tick_params(direction = 'in')

# raw data. Use same x axis, other y
ax2 = ax.twinx()
line2 = ax2.plot(raw_normalized, 
         color = 'red',
         label =r'$P/P_0$'
         )
ax2.set_ylabel(r'$P/P_0$', 
               color = 'red')

# legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc = 'upper left')

#%% saving
plt.savefig('exports/LUTplot.pdf', dpi = 300, bbox_inches = 'tight')
