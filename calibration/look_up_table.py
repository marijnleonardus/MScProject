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
#df_813 = pd.read_csv('lut files/slm5953_at785_HDMI.lut', delimiter = ' ', header = None)
#lut_813 = np.array(df_813)

df_820 = pd.read_csv('lut files/lut820_diffractivecalibration.lut', delimiter = ' ', header = None)
lut_820 = np.array(df_820)
 
#%%plotting
#plt.plot(lut_813[:, 1], label = '813.lut')
plt.plot(lut_820[:, 1], label = '820.lut')

plt.grid()
plt.xlabel('grey level 0-255')
plt.ylabel('voltage response [a.u]')
plt.legend()
plt.savefig('lut values.png', dpi = 300)
