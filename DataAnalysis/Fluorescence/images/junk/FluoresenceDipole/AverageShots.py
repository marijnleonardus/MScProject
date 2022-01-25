# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:40:32 2021

@author: Marijn Venderbosch

Script takes as input a dataset of the type "Spooled files_000x.tif", images
from the Andor camera, assuming every image is a picture with atoms from the dipole
(no MOT background pictures)

"""

#%% Imports

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#%% Loading Data

# Data location and define empty lists

data_location = "data/100us/"
file_prefix = "Spooled files_"

# File numbers. Stop_number is the amount of pixel fixels you made 
start_number = 0
stop_number = 40
step_number = 1  

#%% Load tweezer images

def load_image_list(start, stop, step, data_location, file_prefix):
    iteration_list = []
    
    # Load files
    for i in range(start, stop, step):
        im = Image.open(data_location +
                        file_prefix +
                        str(i).zfill(4) +
                        ".tif")
        array = np.array(im)
        iteration_list.append(array)
    return iteration_list

# Load images in order 0,1,2,3, etc.

image_list = load_image_list(start_number,
                            stop_number, 
                            step_number,
                            data_location,
                            file_prefix)

# Average over multiple shots

def average_list(input_list):
    sum_array = np.zeros(input_list[0].shape)
    
    for i in range(len(input_list)):
        array = input_list[i]
        sum_array += array
    
    # Normalize
    sum_array = sum_array / len(input_list)
    return sum_array

average_shot = average_list(image_list)

#%% Plot result

plt.imshow(average_shot)
plt.savefig("exports/100us.png", dpi = 500)
