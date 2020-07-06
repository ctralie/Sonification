#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:03:13 2020

@author: michaeltecce
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import IPython.display as ipd # This is a library that allows us to play audio samples in Jupyter
from sklearn.decomposition import PCA
import skimage # A library for doing some extra stuff with image processing
import scipy.io.wavfile as wavfile # We will use this library to load in audio
import sys
sys.path.append("..")
from SlidingWindow import*

def extractData():
    
    """
    Takes data from Sunspot Data Sheet
    """
    
    data = np.loadtxt("./SSD.txt",dtype = float)
    return data[:,4]

def interpData(arr, fac):
    """
    Interpolates array by fac
    
    arr: ndarray(N)
        Array of data
    fac: float
        Factor by which to resample.
        If > 1, upsampling
        If < 1, downsampling
    """
    N = len(arr)
    x = np.linspace(0, 1, N)
    f = interpolate.interp1d(x,arr,kind='cubic')
    xnew = np.linspace(0, 1, int(fac*N))
    narr = f(xnew)
    return narr   

def goSinGo(arr):
    
    """
    DEPRICATED
    
    Extends Array by duplicating elements sequentially
    Parameters
    ----------
    arr: Array()
        Array to be extended
    """
    
    sM = 10
    narr = np.linspace(0,1,len(arr)*sM)
    tI = 0
    for i in range(len(arr)):
        for j in range(sM):
            narr[tI] = arr[i]
            tI += 1
    return narr

def positive_data_scale(arr,ran):
    
    """
    Scale Integral Array to set Modulation
    Parameters
    ----------
    arr: Array()
        array to be scaled
    ran: Int
        Factor to Scale By
    """
    
    minN = np.min(arr)
    maxN = np.max(arr)
    rangeN = np.absolute(minN) + np.absolute(maxN)
    positive_offset = (ran*rangeN)
    arr = positive_offset + arr-np.min(arr)
    return arr
    
def apply_freq_mod(x,freqs):
    
    """
    Format audio ndarray to a specific frequency ranges
    Parameters
    ----------
    x: ndarray(N)
        Numpy array of audio samples
    freqs: array()
        Array of Logarithmic Spaced Frequency Ranges 
    """
    
    if len(x.shape) == 1:    
        minF = np.min(x)
        maxF = np.max(x)
    
        setSize = maxF - minF
        freqsSize = freqs[1] - freqs[0]
        scaleSize = setSize/freqsSize
        x = x / scaleSize
    
        newminF = np.min(x)
    
        offsetSize = np.absolute(newminF) + freqs[0]
        x += offsetSize
    else:
        for i in range(len(freqs)-1):
            
            minF = np.min(x[:,i])
            maxF = np.max(x[:,i])
            
            setSize = maxF - minF
            freqsSize = freqs[i+1] - freqs[i]
            scaleSize = setSize/freqsSize
            x[:,i] = x[:,i] / scaleSize
            
            newminF = np.min(x[:,i])
    
            offsetSize = np.absolute(newminF) + freqs[i]
            x[:,i] += offsetSize
    
    return x

def save_wavfile(filename, fs, x):
    
    """
    @author: ctralie
    
    Save audio to a wave file
    Parameters
    ----------
    filename: string
        Name of the file you want to save to
    fs: int
        Sample rate
    x: ndarray(N)
        Numpy array of audio samples
    """
    y = x-np.mean(x)
    y = y/np.max(np.abs(y))
    y = np.array(y*32768, dtype=np.int16)
    print(np.min(y))
    print(np.max(y))
    wavfile.write(filename, fs, y)

if __name__ == '__main__':
    A = extractData()
    B = interpData(A)
    C = goSinGo(B)
    D = doSlidingWindow(C)
    E = doDimRedux(D)
    F = applyAmpMod(E)



