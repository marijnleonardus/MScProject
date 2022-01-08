# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:56:46 2021

@author: Marijn Venderbosch

script exports phase mask as .jpg image

Because the Meadowlark software exports as color image with only one channel
(red) used, we only use this first channel

For one plot, include colorbar as well
"""

#%% imports

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% load image

def load_take_red_channel(bmp_file):
  
    img = Image.open(bmp_file)
    array = np.array(img)

    # take only red RGB channel, transpose
    array_greyscale = array[:,:,0]
    return array_greyscale

array = load_take_red_channel('Z04.bmp')

#%% plotting, saving

fig, ax = plt.subplots(figsize = (2, 2))

ax.axis('off')
plot = ax.imshow(array,
          cmap = 'gray')

# colorbar

cbar = fig.colorbar(plot, 
             ax = ax,
             pad = 0.12,
             shrink = 0.9,
             aspect = 11)
cbar.set_ticks([0, 2, 255])

# save plot includingcolor bar
plt.savefig('exports/ZernikePlusColorbar.jpg',
            bbox_inches = 'tight',
            dpi = 300)

# save as image

image_greyscale = Image.fromarray(array)
image_greyscale.save('exports/zernike.pdf')