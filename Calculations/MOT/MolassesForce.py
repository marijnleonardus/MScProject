#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 23:48:23 2021

first plot:
Example PyLCP library from
https://python-laser-cooling-physics.readthedocs.io/en/latest/examples/MOTs/00_F0_to_F1_1D_MOT_forces.html

Plots spatial depenence. 

second plot: plots velocity force components, formula from thesis

@author: marijn
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import pylcp
from pylcp.atom import atom
import scipy.constants as cts
from matplotlib.ticker import MultipleLocator

#%% Units

rb87 = atom('85Rb')

klab = 2 * np.pi * rb87.transition[1].k # Lab wavevector (without 2pi) in cm^{-1}
taulab = rb87.state[2].tau  # Lifetime of 6P_{3/2} state (in seconds)
gammalab = 1 / taulab
Blab = 15 # About 15 G/cm is a typical gradient for Rb

# Now, here are our `natural' length and time scales:
x0 = cts.hbar * gammalab / (cts.value('Bohr magneton') * 1e-4 * 15) # cm
t0 = klab * x0 * taulab # s
mass = 85 * cts.value('atomic mass constant') * (x0 * 1e-2)**2 / cts.hbar / t0

# And now our wavevector, decay rate, and magnetic field gradient in these units:
k = klab * x0
gamma = gammalab * t0
alpha = 1.0 * gamma     # The magnetic field gradient parameter

#%% Hamiltonian, field, lasers

Hg, mugq = pylcp.hamiltonians.singleF(F = 0, muB = 1)
He, mueq = pylcp.hamiltonians.singleF(F = 1, muB = 1)

dijq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)

ham = pylcp.hamiltonian(Hg,
                        He, 
                        mugq, 
                        mueq, 
                        dijq,
                        mass = mass,
                        gamma = gamma,
                        k = k)

det = -1.5
s = 2

# Define the laser beams:
    
laserBeams = pylcp.laserBeams(
    [{'kvec':k*np.array([1., 0., 0.]), 
      's':s,
      'pol':-1, 
      'delta':det * gamma},
     {'kvec':k*np.array([-1., 0., 0.]),
      's':s,
      'pol':-1, 
      'delta':det * gamma}],
    beam_type=pylcp.infinitePlaneWaveBeam
)

# Define the magnetic field:
linGrad = pylcp.magField(lambda R: -alpha*R)

rateeq = pylcp.rateeq(laserBeams,
                      linGrad, 
                      ham, 
                      include_mag_forces = True)
#heuristiceq = pylcp.heuristiceq(laserBeams, linGrad, gamma=gamma, k=k, mass=mass)

#%% Force profile

x = np.arange(-30, 30, .1) / (alpha / gamma)
v = np.arange(-30, 30, .1)

X, V = np.meshgrid(x, v)

Rvec = np.array([X, np.zeros(X.shape), np.zeros(X.shape)])
Vvec = np.array([V, np.zeros(V.shape), np.zeros(V.shape)])

rateeq.generate_force_profile(Rvec,
                              Vvec,
                              name='Fx',
                              progress_bar = True)
#heuristiceq.generate_force_profile(Rvec, Vvec, name='Fx', progress_bar=True)

#%% Velocity profile, formula from thesis

plot_range = 5
data_points = 100

omega_d = np.linspace(-plot_range, plot_range, data_points)

def forces(detuning, S, omega_d):
    force_plus = 0.5 * S / (1 + S + 4 * (detuning - omega_d)**2)
    force_minus=-0.5 * S / (1 + S + 4 * (detuning + omega_d)**2)

    force_total = force_plus + force_minus
    
    return force_plus, force_minus, force_total

force_plus, force_minus, force_total = forces(det, s, omega_d)


#%% Plotting 

fig, (ax2, ax1) = plt.subplots(nrows=1, 
                               ncols=2, 
                               num="Expression",
                               figsize=(6.5, 2.75),
                               sharey = True)

ax1.grid()

ax1.plot(x * (alpha / gamma), 
         rateeq.profile['Fx'].F[0, int(np.ceil(x.shape[0]/2)), :] / gamma / k)

# Heuristic equation plot, commented for now
#ax1.plot(x*(alpha/gamma), heuristiceq.profile['Fx'].F[0, int(np.ceil(x.shape[0]/2)), :]/gamma/k)
#ax1.plot(x*(alpha/gamma), (s/(1+2*s + 4*(det-alpha*x/gamma)**2) - s/(1+2*s + 4*(det+alpha*x/gamma)**2))/2, 'k-', linewidth=0.5)


ax2.set_ylabel(r'$F [\hbar k \gamma$]', usetex = True, family = 'serif')
ax1.set_xlabel('$z \\ [\mu_B (\partial B/\partial z) / \hbar \gamma]$', usetex = True)
ax1.set_xlim((-10, 10))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(.1))


# Second plot: velocity profile

ax2.grid()

ax2.plot(omega_d, force_plus, 
        label = r'$F^{+}$',
        linewidth = 1.5,
        linestyle = '--',
        color = 'blue')

ax2.plot(omega_d, force_minus,
        label = r'$F^{-}$',
        linewidth = 1.5,
        linestyle = '--',
        color = 'red')

ax2.plot(omega_d, force_total,
        label = r'$F^{+}+F^{-}$',
        linewidth = 1.5,
        linestyle = '-',
        color = 'purple')

plt.savefig('SpatialForceMOT.pdf',
            dpi = 200, 
            pad_inches = 0,
            bbox_inches = 'tight')

ax2.set_xlabel(r'$v [\gamma/k$]', usetex = True, family = 'serif')
ax2.xaxis.set_minor_locator(MultipleLocator(1))

ax2.set_xlim(-5, 5)

ax2.legend()

# Saving

plt.savefig('exports/SpatialVelocityDependence.pdf', 
            dpi = 200, 
            pad_inches = 0,
            bbox_inches = 'tight')
