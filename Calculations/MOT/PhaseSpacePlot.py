#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:29:35 2021

Script only slightly editted but mainly copied from library 
https://python-laser-cooling-physics.readthedocs.io/

Particular example 
https://python-laser-cooling-physics.readthedocs.io/en/latest/examples/MOTs/01_F0_to_F1_1D_MOT_capture.html?highlight=trajectories#Add-trajectories-in-phase-space

Script plots phase space around equilibrium point of Rb85 MOT
@author: marijn
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
from scipy.optimize import bisect #For root finding
import pylcp
import pylcp.atom as atom
from pylcp.common import progressBar

#%% Units

x0 = (6 / 1.4 / 15) # cm
k = 2 * np.pi / 780e-7 # cm^{-1}
kbar = k * x0

gamma = 2 * np.pi * 6.0666e6
t0 = k * x0 / gamma

mass = 84.911 * cts.value('atomic mass constant') / (cts.hbar * (k * 1e2)**2 * t0)

#%% Laser beam, field, Hamiltonian

det = -1.5
alpha = 1.0
s = 2

laserBeams = pylcp.laserBeams([
    {'kvec':np.array([0., 0., 1.]),
     'pol':np.array([0., 0., 1.]), 
     's':s, 
     'delta':det},
    {'kvec':np.array([0., 0., -1.]), 
     'pol':np.array([1., 0., 0.]), 
     's':s, 
     'delta':det}],
    beam_type = pylcp.infinitePlaneWaveBeam
)

magField = pylcp.quadrupoleMagneticField(alpha)

# Use the heuristic equation (or comment it out):
eqn = pylcp.heuristiceq(laserBeams,
                        magField,
                        gamma = 1,
                        mass = mass)

#%% Equilibrium force

dz = 0.1
dv = 0.1

z = np.arange(-20, 20 + dz, dz)
v = np.arange(-20, 20 + dv, dv)

Z, V = np.meshgrid(z, v)

Rfull = np.array([np.zeros(Z.shape), np.zeros(Z.shape), Z])
Vfull = np.array([np.zeros(Z.shape), np.zeros(Z.shape), V])

eqn.generate_force_profile([np.zeros(Z.shape), np.zeros(Z.shape), Z],
                           [np.zeros(V.shape), np.zeros(V.shape), V],
                           name = 'Fz',
                           progress_bar = True);


#%% Trajectories phase space

v0s = np.linspace(-1, 13, 14)

# See solve_ivp documentation for event function discussion:
def captured_condition(t, y, threshold = 1e-5):
    if(y[-4] < threshold and y[-1] < 1e-3):
        val = -1.
    else:
        val = 1.

    return val

def lost_condition(t, y, threshold = 1e-5):
    if y[-1] > 20.:
        val = -1.
    else:
        val = 1.

    return val

captured_condition.terminal = True
lost_condition.terminal = True

sols = []
for v0 in v0s:
    eqn.set_initial_position_and_velocity(np.array([0., 0., z[0]]),
                                          np.array([0., 0., v0]))
    if isinstance(eqn, pylcp.rateeq):
        eqn.set_initial_pop(np.array([1., 0., 0., 0.]))

    eqn.evolve_motion([0., 100.], 
                      events = [captured_condition, lost_condition], max_step=0.1)

    sols.append(eqn.sol)

#%% Plot Trajectories

fig, ax = plt.subplots(1, 1, figsize = (5,4))

# Lines through origin

plt.axvline(x = 0, 
            color = 'grey',
            linewidth = 0.85)

plt.axhline(y = 0, 
            color = 'grey', 
            linewidth = 0.85)


plt.imshow(2 * eqn.profile['Fz'].F[2], 
           origin = 'lower',
           extent = (np.amin(z)-dz/2, np.amax(z)-dz/2,
                   np.amin(v)-dv/2, np.amax(v)-dv/2),
           aspect = 'auto',
           cmap = 'bwr')

# Colorbar

cb1 = plt.colorbar()
cb1.set_label('$F \\ [\gamma / \hbar k]$', usetex = True)

ax.set_xlabel('$x \\ [\mu_B \partial B/\partial z) / \hbar \gamma]$', usetex = True)
ax.set_ylabel('$v \\ [\gamma/k]$', usetex = True)

#fig.subplots_adjust(left = 0.15, right = 0.91, bottom = 0.2)

for sol in sols:
    ax.plot(sol.r[2], sol.v[2],
            color = 'black', 
            linewidth = 1)

#ax.yaxis.set_ticks([-20, -10, 0, 10, 20])
# Display the figure at the end of the thing.
ax.set_xlim((-10, 10))
ax.set_ylim((-12, 12))

plt.savefig('exports/PhaseSpace.pdf',
            dpi = 200, 
            pad_inches = 0,
            bbox_inches = 'tight')


