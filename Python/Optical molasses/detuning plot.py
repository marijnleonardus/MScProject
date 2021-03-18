# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:41:50 2021

@author: Marijn Venderbosch
"""


import matplotlib.pyplot as plt
import numpy as np

# We make the problem dimensionless
hbar = 1
k = 1
gamma = 1

# The saturation parameter can be varied, here a couple examples:
saturation = [1/8 , 1/4, 1/2, 1, 2]

# Detuning is our independent parameter
detuning = np.linspace(-3 , -0.01, 1000)

# Plotting for several saturation paramters
for i in saturation:
    damping_coefficient = -8 * i * detuning / gamma * 1 / (1 + i + (2 * detuning / gamma)**2)**2
    plt.plot(detuning , damping_coefficient , label='s0 = '+ str(i))
    
# Reverse order x axis because Detuning <0 for a red detuned trap
plt.axis([max(detuning) , min(detuning) , min(damping_coefficient) , max(damping_coefficient)])

plt.ylim(0 , 0.55)
plt.grid()
plt.xlabel(r'Detuning $\delta$ [$\gamma$]')
plt.ylabel(r'Damping coefficient $\beta$ [$\hbar k^2$]')
plt.legend()

# Saving
plt.savefig('Detuning plot.png' , dpi = 500)