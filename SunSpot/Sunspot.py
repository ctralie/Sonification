#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:03:13 2020

@author: michaeltecce
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from SlidingWindow import*
from sklearn.decomposition import PCA


def extractData():
    data = np.loadtxt("./SSD.txt",dtype = float)
    return data[:,4]

def interpData(A):
    x = np.linspace(0, 1, len(A))
    f = interpolate.interp1d(x,A)
    B = f(x)
    return B   

def doSlidingWindow(B):
    Tau = 10
    dim = 40
    dT = 1
    x = np.cos(B)
    C = getSlidingWindow(x,dim,Tau,dT)
    return C

def doDimRedux(C):
    pca = PCA(n_components = 10)
    D = pca.fit_transform(C)
    return D

def applyAmpMod(D):
    W = 1
    E = np.zeros(D.shape[0])
    for i in range(D.shape[1]):
        E += np.sin(2*np.pi*D[:,i])
        W += 1
    return E
    

A = extractData()
B = interpData(A)
C = doSlidingWindow(B)
D = doDimRedux(C)
E = applyAmpMod(D)



