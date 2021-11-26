# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:56:46 2021

@author: Marijn Venderbosch

SCript exports phase mask as .png image

Because the Meadowlark software exports as color image with only one channel
(red) used, we only use this first channel

For one plot, include colorbar as well
"""

#%% imports

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% load image

img = Image.open('superimposed.bmp')
array = np.array(img)

# take only red RGB channel, transpose
array_greyscale = np.transpose(
    array[:,:,0])


#%% plotting
fig, ax = plt.subplots(figsize = (2, 2))

ax.axis('off')
plot = ax.imshow(array_greyscale,
          cmap = 'gray')

# colorbar

cbar = fig.colorbar(plot, 
             ax = ax,
             pad = 0.12,
             shrink = 0.9,
             aspect = 11)
cbar.set_ticks([0, 127, 255])

# save plot
plt.savefig('ZernikePlusColorbar.jpg',
            bbox_inches = 'tight',
            dpi = 300)


#%% save as image

image_greyscale = Image.fromarray(array_greyscale)
image_greyscale.save('zernike.png')
