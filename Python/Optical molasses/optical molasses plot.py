# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:46:59 2021

@author: Marijn Venderbosch
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters, the problem is made dimensionless by choosing gamma and k unity
gamma = 1
k = 1
hbar = 1

# Parameters that can be changed for this plot
s_0 = 0.5
detuning = -gamma

# Initializing velocity vector ranging from -4 k/gamma to +4 k/gamma
velocity=1 / (gamma / k) * np.linspace(-4, 4, 1000)

# Calculating the Force contributions from both beams and adding them
F_positive = + s_0 / 2 * 1 / (1 + s_0 + 4 * ((detuning - k * velocity) / gamma)**2)
F_negative = - s_0 / 2 * 1 / (1 + s_0 + 4 * ((detuning + k * velocity) / gamma)**2)
F_total = F_positive + F_negative

# Plotting
plt.plot(velocity , F_positive , label = '+x beam')
plt.plot(velocity , F_negative , label = '-x beam')
plt.plot(velocity , F_total , label = 'total')

plt.grid()
plt.legend()
plt.xlabel(r'Velocity [$\gamma /k$]')
plt.ylabel(r'Force [$\gamma \hbar k$]')

# Saving
plt.savefig('optical molasses.png', dpi=500)

