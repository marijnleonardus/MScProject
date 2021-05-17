#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:41:25 2021

@author: marijn
"""

from PIL import Image

im = Image.open('12 5 10x10 averaged grey.png') #open color image
im = im.convert('1') # black/white
im.save('greyscale.png')
print(type(im))