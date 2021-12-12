# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:30:14 2021

@author: Marijn Venderbosch

Script computes amount of power in a tweezer needed as a function of 
    - desired trap depth
    - tweezer waist
    - tweezer wavelength (detuning)
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

#%% constants and variables
# constants
c = const.c # m/s
m_e = const.electron_mass # kg
eps_0 = const.epsilon_0 # A^2 s^4 kg^-1 m^-3
e = const.elementary_charge # A s 
k_b = const.Boltzmann #J/K

# variables
waist0_7 = 0.76e-6
T = 1e-3 #K
transition_wl = 780e-9
laser_wl = 820e-9
standard_trapdepth = 1 # mK

# derived variables
transition_angular = 2 * np.pi * c / transition_wl
laser_angular = 2 * np.pi * c / laser_wl
detuning820 = transition_angular - laser_angular
trap_depth = k_b * T

#%% calculation from mathematica script, substituting the definition
# of power in the equation of the trap depth of an optical dipole trap
# P = U_0 omega^2 detuning omega_0^3 / (3 c Gamma)

# We need the Rabi frequency matrix element, which we estimate from 
# a classical harmonic oscillator model (Grimm, 1999)
matrix_element = transition_angular**2 / laser_angular**2 * e**2 * laser_angular**2 / (6 * np.pi * eps_0 * m_e * c **3)

# Power needed as a function of trap depth [mK], waist, detuning
def power_needed(trap_depth, waist, detuning):
    power = trap_depth * waist**2 * detuning * transition_angular / (3 * c * matrix_element)
    return power

power_standard_parameters = power_needed(standard_trapdepth, waist0_7, detuning820)

#%% plot power vs trap depth

# parameter sweeps
trap_depth_range_mK = np.linspace(0, 3, 101)
waist_range = np.linspace(0.6e-6, 1e-6, 101)
trapping_frequency_range = np.linspace(790e-9, 850e-9, 101)
detuning_range =- 2 * np.pi * c / trapping_frequency_range + transition_angular

# Initialize plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, 
                                    ncols = 3,
                                    figsize = (10, 3))

# Plot power as a function of trap depth for standard waist, detuning used in our experiment
ax1.plot(trap_depth_range_mK, power_needed(trap_depth_range_mK,
                                           waist0_7, 
                                           detuning820))
ax1.grid()
ax1.set_xlabel(r'Trap depth $U_0/k_b$ [mK]')
ax1.set_ylabel(r'P [mW]')
ax1.scatter(standard_trapdepth, 
            power_standard_parameters,
            s = 30,
            color = 'red')

# Plot power as a function of waist, using standard trap depth and detuning
ax2.plot(waist_range, power_needed(standard_trapdepth,
                                   waist_range, 
                                   detuning820))
ax2.grid()
ax2.set_xlabel(r'Waist $w$ [$\mu$m]')
ax2.scatter(waist0_7,
            power_standard_parameters,
            s = 30,
            color = 'red')

# Plot power as function of laser frequency (detuning) using standard trap depth, waist. 
# Scale x axis by 10^-9 to get in nm
ax3.plot(trapping_frequency_range / 1e-9, power_needed(standard_trapdepth, 
                                                       waist0_7,
                                                       detuning_range))
ax3.grid()
ax3.set_xlabel(r'Wavelength $\lambda$ [nm]')
ax3.scatter(laser_wl/1e-9,
            power_standard_parameters,
            s = 30,
            color = 'red')

plt.savefig('Trapdepth_vs_power_tweezer.pdf',
            dpi = 300,
            bbox_inches = 'tight')



