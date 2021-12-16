# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 09:54:40 2021

@author: Marijn Venderbosch

Computes trap depth from eq. 1.6 PhD thesis Ludovic Brossard
"""

#%% Imports

from scipy.constants import Boltzmann, c, pi, hbar, Planck

#%% Variables

# Dipole trap
power = 3.5e-3 #mW
waist = 1e-6 # m
wavelength = 850e-9 # m

# D1 Rubidium
linewidth_D1 = 2 * pi * 5.7e6 # 1/s
transition_wavelength_D1 = 795e-9 # m

# D2 Rubidium
linewidth_D2 = 2 * pi * 6e6 # 1/s
transition_wavelength_D2 = 780e-9

# Functions

def intensity(power, waist):
    return 2 * power / (pi * waist**2)

def detuning(wavelength, transition_wavelength):
    return 2 * pi * c * (1 / wavelength - 1 / transition_wavelength)

def saturation_intensity(linewidth, transition_wavelength):
    return 2 * pi**2 * hbar * c * linewidth / (3 * transition_wavelength**3)

def dipole_potential(detuning_D1,
                     detuning_D2, 
                     transition_wavelength_D1, 
                     transition_wavelength_D2):
    # eq. 1.6 from Brossard PhD thesis (Browaeys, 2020)
    # matrix elements same for all transitions because linear polarization
    # prefactors 1/3 and 2/3 prefactors come from 2J'+1/2J+1 and Clebsch-Gordon
    
    D1_contribution = 1 * linewidth_D1**2 / (3 * saturation_D1 * detuning_D1)
    D2_contribution = 2 * linewidth_D2**2 / (3 * saturation_D2 * detuning_D2)
    
    return hbar / 8 * (D1_contribution + D2_contribution) * intensity(power, waist)

#%% Executing functions and print result

detuning_D1 = detuning(wavelength, transition_wavelength_D1)
detuning_D2 = detuning(wavelength, transition_wavelength_D2)

saturation_D1 = saturation_intensity(linewidth_D1, transition_wavelength_D1)
saturation_D2 = saturation_intensity(linewidth_D2, transition_wavelength_D2)

dipole_potential = dipole_potential(detuning_D1,
                                    detuning_D2, 
                                    transition_wavelength_D1,
                                    transition_wavelength_D2)

potential_depth_mK = round(-dipole_potential / Boltzmann * 1e3, 2)
print("Trap depth is: " + str(potential_depth_mK) + " mK")

potential_depth_MHz = round(-dipole_potential / Planck * 1e-6, 1)
print("Trap depth (Hz) is: " + str(potential_depth_MHz) + "MHz")


    


