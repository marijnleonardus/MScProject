#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:41:31 2021

@author: marijn

Script makes a plot of the laser induced fluorescence from the MOT with
a color overlay
"""

# Imports
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import unravel_index

# Variables
window_size = 110      

# Import .bmp
image = Image.open('mot.bmp')
array = np.array(image) 

# Finding center MOT
max_loc = array.argmax()
indices= unravel_index(array.argmax(), array.shape)

# Cropping                                                     
RoI = array[indices[0] - window_size : indices[0] + window_size, 
            indices[1] - window_size : indices[1] + window_size]
             
# Ploting
img = plt.imshow(RoI)
img.set_cmap('inferno')
plt.axis('off')
plt.savefig('LiF_MOT.png', dpi = 300)
