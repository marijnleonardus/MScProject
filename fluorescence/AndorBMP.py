# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:21:01 2021

@author: marijn

loads bmp file from Andor sCMOS camera

Export image as BMP using GUI application from them
"""

#%% imports

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% load image file

location = 'images/' 
filename  = 'rescaled bmp output.bmp'
bmp_file = Image.open(location + filename)
array = np.array(bmp_file)

#%% plotting and saving

fig, ax = plt.subplots()
ax.imshow(array)
