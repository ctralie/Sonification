#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:37:44 2020

@author: michaeltecce
"""


import numpy as np # This is the main numerical library we will use
import matplotlib.pyplot as plt # This is the main plotting library we will use

import sys
sys.path.append("..")
from CurvatureTools import *

"""
HP 20-16k
cs 70-17k
bs 50-16k
cp 80-17k

h-l = tf
tf/dim=freq ranges
"""



def apply_freq_mod(YIntegDeriv,freqs,decfac,dcom,ST = 0):
    
    """
    ST = speaker type:
        0 = computer speaker
            70hz - 17khz
        1 = headphones
            20hz - 16khz
        2 = cell phone speaker
            80hz - 17khz
        3 = bluetooth speaker
            50hz - 16khz
    
    s(t) = cos(2*pi*f(t))
    frequency = f'(t)
    f(t) = k*t
    s(t) = cos(2*pi*k*t)    
    """
    
    """
    take column from Y
    get derivative
    calc min,max,mean freq
        np.min()
        np.max()
        np.mean()
    
    
    """
    
    sT = np.cos(2*np.pi*YIntegDeriv)
    tD = getCurvVectors(sT, 1, decfac)[1]
    fM = np.zeros((len(YIntegDeriv),dcom))

    for i in range(len(freqs)-1):
        minF = np.min(tD[:,i])*(len(YIntegDeriv))
        maxF = np.max(tD[:,i])*(len(YIntegDeriv))
        
        setSize = np.absolute(minF) + np.absolute(maxF)
        freqsSize = freqs[i+1] - freqs[i] 
        scaleF = setSize / freqsSize
        
        scaleMax = maxF/scaleF
        freqBump = freqs[i+1] - scaleMax
        
        fM[:,i] = np.cos(2*np.pi*(1/scaleF)*YIntegDeriv[:,i]+freqBump)

    return fM

