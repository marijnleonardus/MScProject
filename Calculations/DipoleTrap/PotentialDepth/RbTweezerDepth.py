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
D2wavelength = 780e-9 # m
D1wavelength = 795e-9 # m

omega_D2 = 2 * np.pi * c/ D2wavelength # 1/s
omega_D1 = 2 * np.pi * c/ D1wavelength # 1/s
omega = 2 * np.pi * c / laser_wavelength # 1/s

waist = 1e-6 # m
power = 5e-3 # W


#%% calculations

damping_rate = elementary_charge**2 * omega_D2**2 / (6 * np.pi * epsilon_0 * electron_mass * c**3)
detD2 = omega - omega_D2
detD1 = omega- omega_D1

intensity = 2 * power / (np.pi * waist**2)

potential_depth = Planck / (2 * np.pi) / 8 * damping_rate**2 / 1.3e-2 *(1 / (3 * detD1) + 2 / (3 * detD2))

potential_depth_mK = round(potential_depth / Boltzmann * 1e3, 1)
potential_Hz = round(potential_depth / Planck * 1e-6, 2)

print("Dipole potential is: "+ str(potential_depth_mK)+ "mK or " + str(potential_Hz)+ " MHz deep")