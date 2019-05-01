# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:38:03 2019

@author: aria
"""
import numpy as np

#define grid spacing
#DELTA_T = 0.05 #this is the value that worked for part 1
DELTA_T = 0.1
ETA = 0.000002
VPRIME = 0.1#this value is toggled for the different parts
#define boundaries
LOW = -np.pi
HIGH = np.pi
#define physical stuff
CHARGE = 1.0