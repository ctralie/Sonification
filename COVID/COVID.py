#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:37:50 2020

@author: michaeltecce
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy import interpolate
import sys
sys.path.append("..")
sys.path.append('../Viterbi')
from SlidingWindow import *
from Viterbi import getCSM

def get_data(countrycode):

    data = pd.read_csv('COVIDDATA.csv')
    data = data.to_numpy()
    
    notC = True
    startindex = 0
    while notC:
        if data[startindex,1] == countrycode:
            notC = False
        else:
            startindex += 1
            
    isC = True
    endindex = startindex
    while isC:
        if data[endindex,1] == countrycode:
            endindex += 1
        else:
            isC = False
    
    return data[startindex:endindex]

def get_info(Data):
    cases = (Data[:,4]).astype(int)
    cases = make_positive(cases)
    deaths = (Data[:,6]).astype(int)
    deaths = make_positive(deaths)
    return cases, deaths

def make_positive(Data):
    for i in range(len(Data)):
        if Data[i] < 0:
            Data[i] *= -1
    return Data

def get_interp(USC,USD,ITC,ITD,DAF):
    IUSC = interp_data(USC,DAF)
    IUSD = interp_data(USD,DAF)
    IITC = interp_data(ITC,DAF)
    IITD = interp_data(ITD,DAF)
    return IUSC,IUSD,IITC,IITD

def interp_data(Data, DAF):
    #Interpolate Time Series to new size
    fac = len(DAF)/len(Data)
    N = len(Data)
    x = np.linspace(0, 1, N)
    f = interpolate.interp1d(x,Data,kind='linear')
    xnew = np.linspace(0, 1, int(fac*N))
    return f(xnew)

def create_2D_shape(Cases,Deaths):
    s = len(Cases)
    Shape = np.zeros((s,2))
    Shape[:,0] = Cases[:]
    Shape[:,1] = Deaths[:]
    return Shape
 
def get_country_max(DO,DT):
    domax = np.max(DO)
    dtmax = np.max(DT)
    return domax,dtmax
    
def create_note_grid_arrays(cmax,dmax):
    """
    MAJ7:
        a,e
    MAJ:
        a,f
        b,e
    MIN:
        a,g
        b,f
        c,e
    MIN7:
        a,h
        b,g
        c,f
        d,e
    FDIM:
        b,h
        c,g
        d,f
    DIM:
        c,h
        d,g
    TRI:
       d,h
    """
    a = 0
    b =  cmax/3
    c = 2*cmax/3
    d =  cmax
    e = 0
    f = dmax/3
    g = 2*dmax/3
    h = dmax
    MAJS = np.array([[a, e]])
    MAJ = np.array([[a,f],[b,e]])
    MIN = np.array([[a,g],[b,f],[c,e]])
    MINS = np.array([[a,h],[b,g],[c,f],[d,e]])
    FDIM = np.array([[b,h],[c,g],[d,f]])
    DIM = np.array([[c,h],[d,g]])
    TRI = np.array([[d,h]])
    return MAJS,MAJ,MIN,MINS,FDIM,DIM,TRI

def make_chord_arrays(Seconds,Types):
    
    fs = 44100
    note = np.linspace(0,Seconds,fs*Seconds)
    numchords = int(len(Types))

    one = note * 440 * (2**(0/12))
    majthree = note * 440 * (2**(5/12))
    minthree = note * 440 * (2**(4/12))
    majfive = note * 440 * (2**(7/12))
    minfive = note * 440 * (2**(6/12))
    dimfive = note * 440 * (2**(5/12))
    minseven = note * 440 * (2**(10/12))
    majseven = note * 440 * (2**(11/12))
    dimseven = note * 440 * (2**(9/12))
    disonance = note * 440 * (2**(3/12))
    one = np.sin(2*np.pi*one)
    majthree = np.sin(2*np.pi*majthree)
    minthree = np.sin(2*np.pi*minthree)
    majfive = np.sin(2*np.pi*majfive)
    minfive = np.sin(2*np.pi*minfive)
    dimfive = np.sin(2*np.pi*dimfive)
    minseven = np.sin(2*np.pi*minseven)
    majseven = np.sin(2*np.pi*majseven)
    dimseven = np.sin(2*np.pi*dimseven)
    disonance = np.sin(2*np.pi*disonance)
    
    Chords = np.zeros((len(note),numchords))

    for i in range(numchords):
        if Types[i] == "Major7":
            Chords[:,i] = one + majthree + majfive + majseven
        elif Types[i] == "Major":
            Chords[:,i] = one + majthree + majfive
        elif Types[i] == "Minor": 
            Chords[:,i] = one + minthree + majfive
        elif Types[i] == "Minor7":
            Chords[:,i] = one + minthree + majfive + minseven
        elif Types[i] == "FullyDim":
            Chords[:,i] = one + minthree + dimfive + dimseven
        elif Types[i] == "Dim":
            Chords[:,i] = one + minthree + dimfive
        elif Types[i] == "Tritone":
            Chords[:,i] = one + disonance + minfive + dimseven
    return Chords

def amp_mod_audio(dists,AS,modamp):
    
    Audio = AS
    x = 0
    
    for i in range(len(dists[x])):
        dfac = len(AS)/len(dists)
        dN = len(dists)
        dx = np.linspace(0, 1, dN)
        fd = interpolate.interp1d(dx,dists[:,i],kind='cubic')
        dxnew = np.linspace(0, 1, int(dfac*dN))
        D = fd(dxnew)
        
        #apply amplitude modulation (I used exponential because simple multiplication didn't have a usable effect)
        ampchange = 1.0 + modamp
        Audio += AS[:] / (D[:]**ampchange)
        x += 1
        
    return Audio

def create_audio(Data,MAJS,MAJ,MIN,MINS,FDIM,DIM,TRI,Chords):
    
    modamp = .25
    
    MAJS = getCSM(Data, MAJS)
    AMJS = amp_mod_audio(MAJS,Chords[:,0],modamp)
    
    MAJ = getCSM(Data, MAJ)
    AMJ = amp_mod_audio(MAJ,Chords[:,1],modamp)
    
    MIN = getCSM(Data, MIN)
    AMN = amp_mod_audio(MIN,Chords[:,2],modamp)
    
    MINS = getCSM(Data, MINS)
    AMNS = amp_mod_audio(MINS,Chords[:,3],modamp)
    
    FDIM = getCSM(Data, FDIM)
    AFD = amp_mod_audio(FDIM,Chords[:,4],modamp)
    
    DIM = getCSM(Data, DIM)
    ADM = amp_mod_audio(DIM,Chords[:,5],modamp)
    
    TRI = getCSM(Data, TRI)
    ATT = amp_mod_audio(TRI,Chords[:,6],modamp)
    
    Audio = AMJS + AMJ + AMN + AMNS + AFD + ADM + ATT
    return Audio    
    