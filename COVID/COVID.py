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
    
    """
    country code = 2 char string for country
    returns said countries data
    """
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
    
    """
    data = matrix of data
    returns the cases and deaths per day 
    """
    
    cases = (Data[:,4]).astype(int)
    cases = make_positive(cases)
    deaths = (Data[:,6]).astype(int)
    deaths = make_positive(deaths)
    return cases, deaths

def make_positive(Data):
    
    """
    takes virus data matrix
    returns the data made positive (to correct errors in the data)
    """
    
    for i in range(len(Data)):
        if Data[i] < 0:
            Data[i] *= -1
    return Data

def get_interp(USC,USD,ITC,ITD,DAF):
    
    """
    interpolates the data to a new size
    """
    
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
    
    """
    makes a 2D array of cases and deaths in one matrix
    """
    
    s = len(Cases)
    Shape = np.zeros((s,2))
    Shape[:,0] = Cases[:]
    Shape[:,1] = Deaths[:]
    return Shape
 
def get_country_max(X):
    
    """
    gets largest value in matricies
    """
    
    cmax = np.max(X[:,0])
    dmax = np.max(X[:,1])
    return cmax,dmax

def scale_data(Data):
    
    """
    scales data down to 0-1 scale
    """
    
    cmax,dmax = get_country_max(Data)
    Data[:,0] /= cmax
    Data[:,1] /= dmax
    return Data
    
def create_earcons(cmax,dmax):
    """
    MAJ8:
        a,d
    MAJ7:
        b,d
    MAJ:
        c,d
    MIN7:
        a,e
    DIM:
        b,e
    AUG:
        c,e
    MIN:
        a,f
    FDIM:
        b,f
    TRI7:
        c,f
    """
    a = 0
    b = cmax/2
    c = cmax
    d = 0
    e = dmax/2
    f = dmax
    
    MAJW = ([a,d])
    MAJS = ([b,d])
    MAJ = ([c,d])
    MINS = ([a,e])
    DIM = ([b,e])
    AUG = ([c,e])
    MIN = ([a,f])
    FDIM = ([b,f])
    TRI = ([c,f])
    earcons = np.array([MAJW,MAJS,MAJ,MINS,DIM,AUG,MIN,FDIM,TRI])
    return earcons

def make_chord_arrays(Seconds,Types,freq):
    
    fs = 44100
    note = np.linspace(0,Seconds,fs*Seconds)
    numchords = int(len(Types))

    base = note * freq * (2**(0/12))
    three = note * freq * (2**(3/12))
    four = note * freq * (2**(4/12))
    six = note * freq * (2**(6/12))
    seven = note * freq * (2**(7/12))
    eight = note * freq * (2**(8/12))
    nine = note * freq * (2**(9/12))
    ten = note * freq * (2**(10/12))
    eleven = note * freq * (2**(11/12))
    twelve = note * freq * (2**(12/12))
    
    base = np.sin(2*np.pi*base)
    three = np.sin(2*np.pi*three)
    four = np.sin(2*np.pi*four)
    six = np.sin(2*np.pi*six)
    seven = np.sin(2*np.pi*seven)
    eight = np.sin(2*np.pi*eight)
    nine = np.sin(2*np.pi*nine)
    ten = np.sin(2*np.pi*ten)
    eleven = np.sin(2*np.pi*eleven)
    twelve = np.sin(2*np.pi*twelve)
    
    Chords = np.zeros((len(note),numchords))

    for i in range(numchords):
        if Types[i] == "MajorW":
            Chords[:,i] = base + four + seven + twelve
        elif Types[i] == "Major7":
            Chords[:,i] = base + four + seven + eleven
        elif Types[i] == "Major": 
            Chords[:,i] = base + four + seven
        elif Types[i] == "Minor7":
            Chords[:,i] = base + three + seven + ten
        elif Types[i] == "Dim":
            Chords[:,i] = base + three + six
        elif Types[i] == "Aug":
            Chords[:,i] = base + four + eight
        elif Types[i] == "Min":
            Chords[:,i] = base + three + seven
        elif Types[i] == "FullyDim":
            Chords[:,i] = base + three + six + ten
        elif Types[i] == "Tri":
            Chords[:,i] = base + three + six + nine
            
    return Chords

def amp_mod_audio(dists,AS,modamp):
    
    #interpolate distance to size of mod audio sample
    dfac = len(AS)/len(dists)
    dN = len(dists)
    dx = np.linspace(0, 1, dN)
    fd = interpolate.interp1d(dx,dists,kind='cubic')
    dxnew = np.linspace(0, 1, int(dfac*dN))
    D = fd(dxnew)
    
    #apply amplitude modulation (I used exponential because simple multiplication didn't have a usable effect)
    ampchange = 1.0 + modamp
    Audio = AS[:] / (D[:]**ampchange)
    
    for i in range(len(Audio)):
        if Audio[i] < 0:
            Audio[i] = 0
    
    return Audio

def create_audio(Data,Earcons,Chords,modamp):
    
    Audio = 0
    Dists = getCSM(Data, Earcons)
    for i in range(len(Earcons)):
        newaudio = amp_mod_audio(Dists[:, i],Chords[:,i],modamp)
    
    return Audio   
    
    #return AMJS,AMJ,AMN,AMNS,AFD,ADM,ATT
if __name__ == '__main__':    
    #covid data from january 20th to july 10th 2020
    UScode = "US"
    ITcode = "IT"
    USData = get_data(UScode)
    ITData = get_data(ITcode)
    
    #get deaths and cases from each country
    USC,USD = get_info(USData)
    ITC,ITD = get_info(ITData)
    
    #interpolate
    fs = 44100
    seconds = 10
    DesiredAudioFrame = np.linspace(0, seconds, int(fs*seconds))
    IUSC,IUSD,IITC,IITD = get_interp(USC,USD,ITC,ITD,DesiredAudioFrame)
    
    USShape = create_2D_shape(IUSC,IUSD)
    ITShape = create_2D_shape(IITC,IITD)
    
    USShape = scale_data(USShape)
    ITShape = scale_data(ITShape)
    
    uscmax,usdmax = get_country_max(USShape)
    itcmax,itdmax = get_country_max(ITShape)
    
    Types = ["MajorW","Major7","Major","Minor7","Dim","Aug","Minor","FullyDim","Tri"]
    freq = 440
    Chords = make_chord_arrays(seconds,Types,freq)
    
    USEarcons = create_earcons(uscmax,usdmax)
    ITEarcons = create_earcons(itcmax,itdmax)
    
    modamp = 1
    USAudio = create_audio(USShape,USEarcons,Chords,modamp)
    ITAudio = create_audio(ITShape,ITEarcons,Chords,modamp)