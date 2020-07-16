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

def get_interp(C,D,DAF):
    
    """
    interpolates the data to a new size
    """    
    IC = interp_data(C,DAF)
    ID = interp_data(D,DAF)
    return IC,ID

def interp_data(Data, DAF):
    #Interpolate Time Series to new size
    N = len(Data)
    x = np.linspace(0, 1, N)
    f = interpolate.interp1d(x,Data,kind='linear')
    xnew = np.linspace(0, 1, len(DAF))
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
    a = cmax/10000
    b = cmax/2
    c = cmax
    d = dmax/10000
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
    sins = [np.sin(2*np.pi*note*freq*(2**(i/12))) for i in range(13)]    
    Chords = np.zeros((len(note),numchords))
    '''
    chords = {'MajorW':[0, 4, 7, 12], 'Major7':[0, 4, 7, 11]}
    for i, typ in enumerate(Types):
        indices = chords[typ]
        # Loop through and sum sinusoids at these indices, and store away
    '''
    for i in range(numchords):
        if Types[i] == "MajorW":
            Chords[:,i] = sins[0] + sins[4] + sins[7] + sins[12]
        elif Types[i] == "Major7":
            Chords[:,i] = sins[0] + sins[4] + sins[7] + sins[11]
        elif Types[i] == "Major": 
            Chords[:,i] = sins[0] + sins[4] + sins[7]
        elif Types[i] == "Minor7":
            Chords[:,i] = sins[0] + sins[3] + sins[7] + sins[10]
        elif Types[i] == "Dim":
            Chords[:,i] = sins[0] + sins[3] + sins[6]
        elif Types[i] == "Aug":
            Chords[:,i] = sins[0] + sins[4] + sins[8]
        elif Types[i] == "Minor":
            Chords[:,i] = sins[0] + sins[3] + sins[7]
        elif Types[i] == "FullyDim":
            Chords[:,i] = sins[0] + sins[3] + sins[6] + sins[10]
        elif Types[i] == "Tri":
            Chords[:,i] = sins[0] + sins[3] + sins[6] + sins[9]
    return Chords

def amp_mod_audio(dists,AS,modamp):
    #apply amplitude modulation (I used exponential because simple multiplication didn't have a usable effect)
    ampchange = 1.0 + modamp
    Audio = AS[:] / (0.05 + dists[:]**ampchange)
    #Audio = AS[:]*np.exp(-dists[:]**ampchange)
    return Audio

def create_audio(Data,Earcons,Chords,modamp):
    Audio = np.zeros((len(Data)))
    Dists = getCSM(Data, Earcons)
    Dists = Dists/np.max(Dists)    
    for i in range(len(Earcons)):
        newaudio = amp_mod_audio(Dists[:,i],Chords[:,i],modamp)
        Audio += newaudio    
    return Audio

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
    seconds = 1
    DesiredAudioFrame = np.linspace(0, seconds, int(fs*seconds))
    IUSC,IUSD = get_interp(USC,USD,DesiredAudioFrame)
    IITC,IITD = get_interp(ITC,ITD,DesiredAudioFrame)
    
    
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
    
    modamp = .25
    USAudio = create_audio(USShape,USEarcons,Chords,modamp)
    ITAudio = create_audio(ITShape,ITEarcons,Chords,modamp)