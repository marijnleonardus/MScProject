#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 00:01:17 2021

@author: marijn
"""

#%% libraries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#%% load and crop
img = np.array(Image.open('exports/6x6_calc_output.bmp'))

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

img_cropped = crop_center(img, 150, 150)

#%%plot and save
fig, ax = plt.subplots()
ax.imshow(img_cropped)



