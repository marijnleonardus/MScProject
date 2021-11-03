#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage of Blink_C_wrapper.dll
Meadowlark Optics Spatial Light Modulators
"""

#%% Imports
import numpy
from ctypes import c_uint, CDLL, cdll, c_ubyte, POINTER
import ctypes

#%% Load C libraries
"""MAKE SURE THE WINDOW SHOWS UP IN THE WRITE PLACE FOR THE DPI SETTINGS"""
awareness = ctypes.c_int()
errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
print(awareness.value)

# Set DPI Awareness  (Windows 10 and 8)
errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
# the argument is the awareness level, which can be 0, 1 or 2:
# for 1-to-1 pixel control I seem to need it to be non-zero (I'm using level 2)

# Load the DLL
# Blink_C_wrapper.dll, HdmiDisplay.dll, ImageGen.dll, freeglut.dll and glew64.dll
# should all be located in the same directory as the program referencing the
# library
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

# Open the image generation library
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\ImageGen")
image_lib = CDLL("ImageGen")

# indicate that our images are RGB
is_eight_bit_image = c_uint(0);

# Initialize the SDK. The requirements of Matlab, LabVIEW, C++ and Python are different, so pass
# the constructor a boolean indicating if we are calling from C++/Python (true), or Matlab/LabVIEW (flase)
bCppOrPython = c_uint(1);
slm_lib.Create_SDK(bCppOrPython);
print ("Blink SDK was successfully constructed");

#%% define pixel dimensions, LUT
height = c_uint(slm_lib.Get_Height());
width = c_uint(slm_lib.Get_Width());
depth = c_uint(slm_lib.Get_Depth());
center_x = c_uint(width.value//2);
center_y = c_uint(height.value//2);

"""
you should replace linear.LUT with your custom LUT file
but for now open a generic LUT that linearly maps input graylevels to output voltages
Using linear.LUT does NOT give a linear phase response
"""
success = 0;
if (height.value == 1200) and (depth.value == 8):
    success = slm_lib.Load_lut("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\LUT Files\\linear4096.lut");

if success > 0: 
    print ("LoadLUT Successful")	
else:
	print("LoadLUT Failed")	
	
#%%create and write image		
# Create two vectors to hold values for two SLM images
ImageOne = numpy.empty([width.value*height.value * 3], numpy.uint8, 'C');

# Create a blank vector to hold the wavefront correction
WFC = numpy.empty([width.value*height.value * 3], numpy.uint8, 'C');

# Generate phase gradients
VortexCharge = 5;
image_lib.Generate_LG(ImageOne.ctypes.data_as(POINTER(c_ubyte)), WFC.ctypes.data_as(POINTER(c_ubyte)), width.value, height.value, depth.value, VortexCharge, center_x.value, center_y.value, 0);

# Write to SLM
slm_lib.Write_image(ImageOne.ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image);

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK();
