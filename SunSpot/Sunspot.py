#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:03:13 2020

@author: michaeltecce
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.decomposition import PCA
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
    f = interpolate.interp1d(x,A)
    xnew = np.linspace(0, 1, int(fac*N))
    B = f(xnew)
    return B   

def goSinGo(B):
    sM = 10
    C = np.linspace(0,1,len(B)*sM)
    tI = 0
    for i in range(len(sM)):
        for j in range(10):
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

def applyAmpMod(E):
    W = 1
    F = np.zeros(E.shape[0])
    for i in range(E.shape[1]):
        F+= np.sin(2*np.pi*E[:,i])
        W += 1
    return F
    

if __name__ == '__main__':
    A = extractData()
    B = interpData(A)
    C = goSinGo(B)
    D = doSlidingWindow(C)
    E = doDimRedux(D)
    F = applyAmpMod(E)



