#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:08:06 2020

@author: michaeltecce
"""

#%matplotlib qt
import numpy as np
import sys
from scipy import interpolate
sys.path.append("..")
sys.path.append('../Viterbi')
from CurvatureTools import *
from SyntheticCurves import *

def earcon_audio(audarr,earcon,VMag,X,dists,s,modfreq=True,modamp=True):
    
    """
    Function to modulate the frequency and amplitude 
    by proximity to a point in 2D array and speed of a curve on a 2D array
    Parameters
    ----------
    audarr: ndarray()
        audio array
    earcon: array(x,y)
        coordinate point in a 2D array
    X:  ndarray()  
        array containing the 2D shape
    dists: array()
        array of distances form earcon point to every point in X
    s:  int
        number of loops the audio performs
    modfreq:   bool
        bool to determine if the frequency is to be modulated
    modamp:    bool
        bool to determine if the amplitude is to be modulated
    
    """
    fs = 44100
    
    #creating audio loop
    AS = audarr
    for i in range(s-1):
        AS = np.append(AS,audarr)
    
    if modfreq == True:
        
        #interpolate speed to size of audio sample
        fac = len(AS)/len(VMag)
        N = len(VMag)
        x = np.linspace(0, 1, N)
        f = interpolate.interp1d(x,VMag,kind='cubic')
        xnew = np.linspace(0, 1, int(fac*N))
        IS = f(xnew)

        #get cumsum of speed for interpolation
        ISInteg = np.cumsum(IS, axis=0)/fs
        scale = ISInteg[len(ISInteg)-1]
        IS = ISInteg / scale

        #mod audio speed to interpolated speed
        lA = len(AS)
        xA = np.linspace(0, 1, lA)
        fA = interpolate.interp1d(xA,AS,kind='cubic')
        AS = fA(IS)
    
    if modamp == True:
        
        #interpolate distance to size of mod audio sample
        dfac = len(AS)/len(dists)
        dN = len(dists)
        dx = np.linspace(0, 1, dN)
        fd = interpolate.interp1d(dx,dists,kind='cubic')
        dxnew = np.linspace(0, 1, int(dfac*dN))
        D = fd(dxnew)

        #apply amplitude modulation (I used exponential because simple multiplication didn't have a usable effect)
        ampchange = 1.25
        AS = AS[:] / (D[:]**ampchange)

    return AS
