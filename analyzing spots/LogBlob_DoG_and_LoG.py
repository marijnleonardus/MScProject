# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:54:25 2021

@author: Marijn Venderbosch

Script will detect blobs using LoG and DoG algorithms. Marks detected blobs
with circles with radii of the Gaussian kernel that detected the blobs. 
"""

import numpy as np
from skimage.feature import blob_dog, blob_log
import matplotlib.pyplot as plt

# Load image from script 'loadmatSaveCamFrame.py'. 
# Image is cropped to region of interest
image = np.load('cam_frame_array_cropped.npy')

# Use LoG blog detection. 
# Max_sigma is the max. standard deviation of the Gaussian kernel used. 
# Num_sigma the number of intermediate steps in sigma.
# Threshold determines how easily blobs are detected. 
blobs_log = blob_log(image, max_sigma = 30, num_sigma = 10, threshold = 0.1)

# Compute radii in the 3rd column by multipling with sqrt(2)
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

# Same as LoG blob detection but not for DoG (diffrernce of Gaussians)
blobs_dog = blob_dog(image, max_sigma = 30, threshold = 0.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

# Put the results from both DoG and LoG in a single list
blobs_list = [blobs_log, blobs_dog]
colors = ['red', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian']

# Zip the data for the LoG and Dog in a single variable
sequence = zip(blobs_list, colors, titles)

# Make a plot for both LoG and DoG images
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex = True, sharey = True)
ax = axes.ravel()

# Plot original image and overlay with circles on spots where blobs are detected
# Radii or cicles are from the gaussian kernels that detected them

# First sum over the plots to make, in this case 2
for index, (blobs, color, title) in enumerate(sequence):
    ax[index].set_title(title)
    ax[index].imshow(image)
    
    # Next sum over the blobs that are detected per algorithm/plot
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color = color, linewidth = 0.5, fill = False)
        ax[index].add_patch(c)
    ax[index].set_axis_off()

# Save and show
plt.savefig('LoG_vs_DoG.png', dpi = 500, tight_layout = True)
plt.show()
