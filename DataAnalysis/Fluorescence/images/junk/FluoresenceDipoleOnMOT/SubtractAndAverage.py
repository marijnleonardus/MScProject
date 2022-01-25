# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:40:32 2021

@author: Marijn Venderbosch

Script takes as input a dataset of the type "Spooled files_000x.tif", images
from the Andor camera, assumes every odd number for x is a shot with the tweezer off
(just MOT), while every even number is with tweezer on as well. 

Subtracts MOT from tweezer + MOT and averages over multiple attemps
"""

#%% Imports

import numpy as np
from PIL import Image
import cv2 
import matplotlib.pyplot as plt

#%% Loading Data

# Data location and define empty lists

data_location = "U:/KAT1/Images/dipole/MOToffDetuningSweep/"
file_name = "15ms_121mhz"
file_prefix = "/Spooled files_"

# File numbers, step is 2 because only odd/even iterations. Stop_number is the
# amount of pixel fixels you made 
start_number = 4
stop_number = 40
step_number = 2  

#%% Load even and odd files lists, corresponding to tweezer on or off

def load_image_list(start, stop, step, data_location, file_prefix):
    iteration_list = []
    
    # Load files
    for i in range(start, stop, step):
        im = Image.open(data_location +
                        file_name + 
                        file_prefix +
                        str(i).zfill(4) +
                        ".tif")
        array = np.array(im)
        iteration_list.append(array)
    return iteration_list

# Even numbers sequence: 0,2,4
even_list = load_image_list(start_number,
                            stop_number, 
                            step_number,
                            data_location,
                            file_prefix)

# Odd numbers start one number later, so 1,3,5
odd_list =  load_image_list(start_number + 1,
                            stop_number,
                            step_number,
                            data_location,
                            file_prefix)

    
#%% Subtract background, average

# Subtract background

def subtract_arrays(even_list, odd_list):
    difference_list = []
    
    for i in range(len(even_list)):
        even_array = np.array(even_list[i])
        odd_array = np.array(odd_list[i])
        
        difference = cv2.absdiff(even_array, odd_array)
        difference_list.append(difference)
    return difference_list

difference_list = subtract_arrays(even_list, odd_list)

# Average over multiple shots

def average_list(input_list):
    sum_array = np.zeros(input_list[0].shape)
    
    for i in range(len(input_list)):
        array = input_list[i]
        sum_array += array
        
        # Normalize
        sum_array = sum_array / len(input_list)
    return sum_array

average_shot = average_list(difference_list)

#%% Plot result

plt.imshow(average_shot)
plt.savefig("exports/" + file_name + ".png",
            dpi = 500)
