#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 19:14:03 2021

"""

#%% imports

from numpy import pi
import math
import scipy.constants as fc

#%% units

MHz = 2*pi*1e6 # rad/s
kHz = 2*pi*1e3 # rad/s
Hz = 2*pi # rad/s

mum = 1e-6 # m
nm = 1e-9 # m
mm = 1e-3 # m
cm2 = 1e-4 # m^2

muW = 1e-6 # W
mW = 1e-3 # W
kW = 1e3 # W
MW = 1e6 # W

Gauss = 1e-4 # T

#%% Sr88 data

m88 = 87.9056121*fc.value('atomic mass constant') # en.wikipedia.org/wiki/Isotopes_of_strontium

trans689 = {'comment':'red MOT', 'wavelength':689*nm,'gamma': 4.7e4}

trans698 = {'comment':'clock',   'wavelength':698*nm,'alpha':198*Hz/math.sqrt(mW/cm2)}
# Omega in rad/s = alpha |B| sqrt(I), B in T and I in W/m^2

trans316 = {'comment':'Rydberg', 'wavelength':316*nm,'gamma': 2*kHz}
# careful: EDM of the transition does not follow from gamma

trans461 = {'comment':'blue MOT','wavelength':461*nm,'gamma': 30*MHz}

trans813 = {'comment':'tweezer', 'wavelength':813*nm,'polarizability':46.597*(kHz/Hz)/(kW/cm2)}
# polarizability from Endres paper PHYS. REV. X 9, 041052 (2019)
# potential depth in Hz is alpha I, I in W/m^2, PHYSICAL REVIEW A 91, 052503 (2015)

trans813 = {'comment':'tweezer', 'wavelength':813*nm,'polarizability':2.644e-3}
# polarizability from Boyd's thesis Table 4.3
# potential depth in Hz is alpha I, I in W/m^2

#%% functions

def angularFrequency(wavelength):
    return fc.speed_of_light/wavelength*Hz

def laserPower(intensity,sigma): # assumes Gaussian intensity profile
    return intensity*2*pi*sigma**2
def laserIntensityFromPower(power,sigma): # assumes Gaussian intensity profile
    return power/(2*pi*sigma**2)
def laserIntensityFromRabi(omega,EDM):
    amp = fc.hbar*omega/EDM # electric field amplitude
    return 0.5*fc.speed_of_light*fc.epsilon_0*amp**2
def saturationIntensity(wavelength,gamma): # M&S 2.24c
    return pi*fc.Planck*fc.speed_of_light*gamma/(3*wavelength**3)

def dipoleMoment(wavelength,gamma): # M&S 2.16b
    omega = angularFrequency(wavelength) # transition frequency
    return math.sqrt(gamma*3*pi*fc.epsilon_0*fc.hbar*(fc.speed_of_light/omega)**3)
def dipoleMomentFromRabi(omega,intensity):
    amp = math.sqrt(intensity/(0.5*fc.speed_of_light*fc.epsilon_0)) # electric field amplitude
    return fc.hbar*omega/amp
def gammaFromEDM(wavelength, EDM): # M&S 2.16b
    omega = 2*pi*fc.speed_of_light/wavelength # transition frequency
    return (omega/fc.speed_of_light)**3*EDM**2/(3*pi*fc.epsilon_0*fc.hbar)
def rabiFromEDM(edm,intensity):
    amp = math.sqrt(intensity/(0.5*fc.speed_of_light*fc.epsilon_0)) # electric field amplitude
    return edm*amp/fc.hbar

def potentialDepthHz(polarizability,intensity):
    return 0.5*polarizability*intensity
def potentialDepthK(polarizability,intensity): # full depth / 1.5 kT
    return potentialDepthHz(polarizability,intensity)*fc.Planck/fc.Boltzmann
def potentialDepthHzFORT(rabi,wavelengthTransition,wavelengthTrap):
    # arxiv.org/pdf/physics/9902072.pdf Eq (10) w.o counter-rotating term
    detuning = angularFrequency(wavelengthTransition)-angularFrequency(wavelengthTrap)
    return rabi**2/(4*detuning)/Hz
def potentialDepthKfromHz(depthHz): # full depth / 1.5 kT
    return (2/3)*depthHz*fc.Planck/fc.Boltzmann

def scatteringRate(wavelength,intensity,transition):
    delta = angularFrequency(wavelength)-angularFrequency(transition['wavelength'])
    gamma = transition['gamma']
    satpar = intensity/(saturationIntensity(transition['wavelength'],gamma))
    return (1/2)*satpar*gamma/(1+satpar+(2*delta/gamma)**2)
def heatingRateK(scatRate,wavelength):
    nrg = 0.5*(fc.Planck/wavelength)**2/m88
    return nrg*scatRate/fc.Boltzmann

def trappingFrequency(potDepthHz,sigma):
    return math.sqrt(fc.Planck*potDepthHz/(m88*sigma**2))
def oscillatorLength(trappingF):
    return math.sqrt(fc.hbar/(trappingF*m88))

#%% check Browaeys parameters

# calculate trap parameters
# should find: depth 1 mK, frad = 100 kHz, flong = 20 kHz
# polarizability and equation for dynamic polarizability from Steck
# unfortunately, answers are not correct
gsPolE = 0.0794*cm2 # Hz/((V/m^2))^2 0.122306 0.0794
alphaStat = gsPolE/(fc.speed_of_light*fc.epsilon_0)
alphaDyn = alphaStat/(1-(780/850)**2) # polarizability in Hz/(W/m^2) for 850 nm trapping
print('alpha',alphaDyn)
trans780 = {'comment':'rubidium', 'wavelength':780*nm,'polarizability':alphaDyn, \
            'gamma': 5.8*MHz}

sigma = .5*mum # think this should really be 1/4 mum
pwr = 3.5*mW
intens = laserIntensityFromPower(pwr,sigma)
depthHz = potentialDepthHz(trans780['polarizability'],intens)
depthK = potentialDepthK(trans780['polarizability'],intens)
scat = scatteringRate(850*nm,intens,trans780)
heat = heatingRateK(scat,850*nm)
trapf = trappingFrequency(depthHz,sigma)
aHO = oscillatorLength(trapf)


print('trap depth',round(depthHz/1e6),'MHz, ',round(depthK*1e3,2),'mK')
#print('scat rate',scat,'1/s, ',heat,'K/s, tau',depthK/heat,'s')
print('trap freq',round(trapf/kHz),'kHz, aHO',round(aHO/nm),'nm')