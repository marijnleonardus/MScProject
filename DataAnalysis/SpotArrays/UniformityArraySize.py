#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 15:34:48 2022

@author: marijn

computes uniformity as a function or array size
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

x = [2, 3, 4, 5, 6, 7, 8]
y = [0, 0, 0, 0, 0, 10.0, 0]

fig, ax = plt.subplots(1,1, figsize = (4,3))
ax.scatter(x,y)

