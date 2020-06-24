#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:03:13 2020

@author: michaeltecce
"""


import numpy as np
import matplotlib.pyplot as plt


def extractData(data):
    lD = len(data)
    A = np.ndarray((lD,1))
    A = data[:,4]
    return A

def applySinWaveGo(A):
    t = np.linspace(0,len(A),len(A))
    for i in range(len(A)):
        t[i] = A[i]
    
    B = np.sin(2*np.pi*440*t)
    return B
    

data = np.loadtxt("./SSD.txt",dtype = float)
A = extractData(data)
B = applySinWaveGo(A)
plt.plot(B)