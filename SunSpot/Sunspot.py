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

def goSinGo(B):
    C = np.linspace(0,1,len(B)*50)
    tI = 0
    for i in range(len(B)):
        for j in range(50):
            C[tI] = B[i]
            tI += 1
    return C        
        
def doSlidingWindow(C):
    Tau = 10
    dim = 40
    dT = 1
    x = np.cos(C)
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
    

A = extractData()
B = interpData(A)
C = goSinGo(B)
D = doSlidingWindow(c)
E = doDimRedux(D)
F = applyAmpMod(E)



