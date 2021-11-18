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

def bmp_import(location, filename):
    bmp_file = Image.open(location + filename)
    array = np.array(bmp_file)
    return array
    
image = bmp_import(location, filename)

#%% plotting and saving

def plot_image(array):
    fig, ax = plt.subplots()
    ax.imshow(image)
