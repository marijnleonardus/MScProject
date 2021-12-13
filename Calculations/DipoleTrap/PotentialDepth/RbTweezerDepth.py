#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:52:29 2021

@author: marijn

Script computes required trap depth for Rb
"""

#%% imports

import numpy as np
from scipy.constants import c, epsilon_0, elementary_charge, electron_mass, Boltzmann, Planck
import scipy.constants as fc

#%% variables

m87 = 84.911789738 * fc.value('atomic mass constant') #kg

laser_wavelength = 820e-9 # m
transition_wavelength = 780e-9 # m

omega_0 = 2 * np.pi * c/ transition_wavelength # 1/s
omega = 2 * np.pi * c / laser_wavelength # 1/s

waist = 1e-6 # m
power = 5e-3 # W


#%% calculations

damping_rate = elementary_charge**2 * omega_0**2 / (6 * np.pi * epsilon_0 * electron_mass * c**3)
detuning = omega - omega_0
intensity = 2 * power / (np.pi * waist**2)

potential_depth = - 3 * np.pi * c**2 / (2 * omega_0**3) * damping_rate * intensity / detuning

potential_depth_mK = round(potential_depth / Boltzmann * 1e3, 1)
potential_Hz = round(potential_depth / Planck * 1e-6, 2)

print("Dipole potential is: "+ str(potential_depth_mK)+ "mK or " + str(potential_Hz)+ " MHz deep")