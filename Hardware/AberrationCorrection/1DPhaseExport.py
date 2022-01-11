
# -*- coding: utf-8 -*-
"""Plots pattern as applied onto SLM in 1D"""

#%% imports

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% variables
coefficient = 0.27

# Initialize pixel array
# Aspect ratio SLM is 1920 / 1200 = 1.6
# Normalize short semi-axis to rho=1 in zernike terms

x = np.linspace(-1.6, 1.6, 1920)
y = np.linspace(-1, 1, 1200)

zernikePlotXData = np.linspace(0, 1, 600)

def ZernikeZ04(coefficient, rho):
    Z04 = 0.26 * (1 - rho**2 + rho**4)
    return Z04

ZernikePlotYData = ZernikeZ04(coefficient, zernikePlotXData)


#%% plotting, saving
plt.plot(zernikePlotXData,ZernikePlotYData)
plt.plot(zernikePlotXData, 0.26 * 6 * zernikePlotXData**4)


