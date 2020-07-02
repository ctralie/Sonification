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
    data = np.loadtxt("./SSD.txt",dtype = float)
    return data[:,4]

def interpData(A, fac):
    """
    A: ndarray(N)
        Array of data
    fac: float
        Factor by which to resample.
        If > 1, upsampling
        If < 1, downsampling
    """
    N = len(A)
    x = np.linspace(0, 1, N)
    f = interpolate.interp1d(x,A,kind='cubic')
    xnew = np.linspace(0, 1, int(fac*N))
    B = f(xnew)
    return B   

def goSinGo(B):
    sM = 10
    C = np.linspace(0,1,len(B)*sM)
    tI = 0
    for i in range(len(B)):
        for j in range(sM):
            C[tI] = B[i]
            tI += 1
    return C
        
def doSlidingWindow(C):
    Tau = 10
    dim = 40
    dT = 1
    x = np.cos(C)*(C/np.max(C))
    D = getSlidingWindow(x,dim,Tau,dT)
    return D

def doDimRedux(D):
    pca = PCA(n_components = 10)
    E = pca.fit_transform(D)
    return E

def positive_data_scale(arr,ran):
    minN = np.min(arr)
    maxN = np.max(arr)
    rangeN = np.absolute(minN) + np.absolute(maxN)
    positive_offset = (ran*rangeN)
    arr = positive_offset + arr-np.min(arr)
    return arr
    
def apply_freq_mod(Y,freqs):
    
    if len(Y.shape) == 1:    
        minF = np.min(Y)
        maxF = np.max(Y)
    
        setSize = np.absolute(minF) + np.absolute(maxF)
        freqsSize = freqs[1] - freqs[0]
        scaleSize = setSize/freqsSize
        Y = Y / scaleSize
    
        newminF = np.min(Y)
    
        offsetSize = np.absolute(newminF) + freqs[0]
        Y += offsetSize
    else:
        for i in range(len(freqs)-1):
            
            minF = np.min(Y[:,i])
            maxF = np.max(Y[:,i])
            
            setSize = np.absolute(minF) + np.absolute(maxF)
            freqsSize = freqs[i+1] - freqs[i]
            scaleSize = setSize/freqsSize
            Y[:,i] = Y[:,i] / scaleSize
            
            newminF = np.min(Y[:,i])
    
            offsetSize = np.absolute(newminF) + freqs[i]
            Y[:,i] += offsetSize
    
    return Y

def save_wavfile(filename, fs, x):
    """
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



