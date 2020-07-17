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
    Returns sect of COVID.csv containing a specified countries data
    Parameters
    ----------
    countrycode:  string
        Country Code used to identify the rows in COVID.csv
    Returns
    -------
    data: ndarray()
        All rows containing the specified country code
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
    Return the Cases per Day & Deaths Per Day of the country's data matrix
    Parameters
    ----------
    Data:  ndarray()
        Data matrix of specific country to be spliced
    Returns
    -------
    cases: array()
        Array of Cases Per Day in chronological order
    deaths:  array()
        Array of Deaths Per Day in chronological order
    """
    cases = (Data[:,4]).astype(int)
    cases = make_positive(cases)
    deaths = (Data[:,6]).astype(int)
    deaths = make_positive(deaths)
    return cases, deaths

def make_positive(Data):
    
    """
    Returns Data with all positive values (to correct errors in COVID data)
    Parameters
    ----------
    Data:  array()
        Array of values to be made positive
    Returns
    -------
    Data:  array()
        Array of values made positive
    """
    for i in range(len(Data)):
        if Data[i] < 0:
            Data[i] *= -1
    return Data

def get_interp(C,D,DAF):
    """
    Returns set of interpolated cases and deaths arrays
    Parameters
    ----------
    C:  array()
        Array of cases to be interpolated
    D:  array()
        Array of deaths to be interpolated
    DAF:  np.linspace()
        Array of set size wished for interpolation of C & D
    Returns
    -------
    IC:  array()
        Interpolated cases array
    ID:  array()
        Interpolated deaths array
    """    
    IC = interp_data(C,DAF)
    ID = interp_data(D,DAF)
    return IC,ID

def interp_data(Data, DAF):
    """
    Returns an interpolated array to size of DAF
    Parameters
    ----------
    Data:  array()
        Array to be interpolated
    DAF:  np.linspace()
        Array of set size used to calculate interpolation
    Returns
    -------
    f(xnew):  array()
        Interpolated Data array
    """
    N = len(Data)
    x = np.linspace(0, 1, N)
    f = interpolate.interp1d(x,Data,kind='linear')
    xnew = np.linspace(0, 1, len(DAF))
    return f(xnew)

def create_2D_shape(Cases,Deaths):
    """
    Returns a 2D array copmrised of cases and deaths
    Parameters
    ----------
    Cases:  array()
        array containing cases per day
    Deaths:  array()
        array containing deaths per day
    Returns
    -------
    Shape:  ndarray()
        2D array, column 1 = cases, column 2 = deaths
    """
    s = len(Cases)
    Shape = np.zeros((s,2))
    Shape[:,0] = Cases[:]
    Shape[:,1] = Deaths[:]
    return Shape
 
def get_country_max(X):
    """
    Returns largest value in each column of a 2D matrix
    Parameters
    ----------
    X:  ndarray()
        Array of Data to be calculated
    Returns
    -------
    cmax:  int
        Max value in 1st column of X
    dmax:  int
        Max value in 2nd column of X
    """
    cmax = np.max(X[:,0])
    dmax = np.max(X[:,1])
    return cmax,dmax

def scale_data(Data,cscale,dscale,scaledown = True):
    """
    Returns scaled data on both columns
    Parameters
    ----------
    Data:  ndarray()
        Array of data to be scaled
    scaledown: bool
        determines if data is scaled down or up
    cscale: float
        how much to multiply column 1 by
    dscale:  float
        how much to multiply column 2 by
    Returns
    -------
    Data:  ndarray()
        Array of scaled data
    """
    if scaledown == True:
        Data[:,0] /= cscale
        Data[:,1] /= dscale
    else:
        Data[:,0] *= cscale
        Data[:,1] *= dscale
    return Data
    
def create_earcons(cmax,dmax,size):
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
    x = np.linspace(cmax/10000,cmax,size)
    y = np.linspace(dmax/10000,dmax,size)
    X,Y = np.meshgrid(x,y)
    E = np.array([X.flatten(), Y.flatten()]).T
    return E

def make_chord_arrays(Seconds,Types,freq):
    """
    Returns array of audio arrays

    Parameters
    ----------
    Seconds: int
        Desired length of audio array to be created
    Types: array(String)
        Array of strings of the type of chord for a specific index
    freq: int
        Desired frequency for chords to be based from

    Returns
    -------
    Chords : ndarray()
        Array of audio arrays

    """
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

def amp_mod_audio(Dists,AS,modamp):
    """
    Returns audio array with amplitude modulation applied

    Parameters
    ----------
    Dists:  array()
        Array of distances used for amplitude modulation
    AS: array()
        Audio array that will have amplitude modulation applied
    modamp: float
        Value to help calculate amount of modulation at a specific distance

    Returns
    -------
    Audio: array()
        Audio array with amplitude modulation applied

    """
    ampchange = 1.0 + modamp
    Audio = AS[:] / (0.01 + Dists[:]**ampchange)
    #Audio = AS[:]*np.exp(-Dists[:]**ampchange)
    return Audio

def create_audio(Data,Earcons,Chords,modamp):
    """
    Returns an audio array with amplitude modulation applied based by the distance
    from a coordinate in Data to an Earcon coordinate

    Parameters
    ----------
    Data : ndarray()
        Cases/Deaths matrix used to calculate distance to earcon
    Earcons : ndarray()
        Array of coordinate points on a 2D grid to calculate distance for amplitude modulation
    Chords : ndarray()
        Array of audio arrays whose amplitude will be modulated based on distance to a specific earcon
    modamp: float
        Value to help calculate amount of modulation at a specific distance

    Returns
    -------
    Audio : array()
        Audio array comprised of all amplitude modulated chords

    """
    Audio = np.zeros((len(Data)))
    Dists = getCSM(Data, Earcons)
    Dists = Dists/np.max(Dists)    
    for i in range(len(Earcons)):
        newaudio = amp_mod_audio(Dists[:,i],Chords[:,i],modamp)
        Audio += newaudio    
    return Audio

def make_casesdeaths_plot(countrycode,Cases,Deaths):
    plt.figure(figsize=(8,8))
    plt.subplot(211)
    plt.plot(Cases)
    plt.ylabel("Cases Per Day")
    plt.xlabel("Days")
    plt.title("New Cases (" + countrycode + ")")
    
    plt.subplot(212)
    plt.plot(Deaths)
    plt.ylabel("Deaths Per Day")
    plt.xlabel("Days")
    plt.title("New Deaths (" + countrycode + ")")
    
def make_data_earcon_plot(Data,Earcons,countrycode,DesiredAudioFrame):
    
    maxc,maxd = get_country_max(Data)
    plt.figure(figsize=(8,8))
    plt.scatter(Data[:, 0], Data[:, 1], c = DesiredAudioFrame)
    plt.scatter(Earcons[:, 0], Earcons[:, 1])
    #plt.xlim(maxc)
    #plt.ylim(maxd)
    plt.xlabel("Cases Per Day")
    plt.ylabel("Deaths Per Day")
    plt.title("Cases/Deaths Earcons Graph of (" + countrycode + ")")

def make_data_earcon_population_plot(Data,Earcons,countrycode,DesiredAudioFrame):
    
    maxc,maxd = get_country_max(Data)
    plt.figure(figsize=(8,8))
    plt.scatter(Data[:, 0], Data[:, 1], c = DesiredAudioFrame)
    plt.scatter(Earcons[:, 0], Earcons[:, 1])
    plt.xlim(0-(maxc*.2),maxc*1.2)
    plt.ylim(0-(maxd*.2),maxd*1.2)
    plt.xlabel("Cases Per Day")
    plt.ylabel("Deaths Per Day")
    plt.title("Cases/Deaths Earcons Graph of (" + countrycode + ")")

def do_covid_calc(countrycode,DesiredAudioFrame,Chords,rcsize,popsize):
    
    #covid data from january 20th to july 10th 2020
    Datamatrix = get_data(countrycode)
    #get deaths and cases from each country
    C,D = get_info(Datamatrix) 
    #make cases deaths graph
    make_casesdeaths_plot(countrycode,C,D)
    
    #interpolate the arrays to desired audio length
    IC,ID = get_interp(C,D,DesiredAudioFrame)
    
    #create 2D arrays from cases/deaths 
    Data = create_2D_shape(IC,ID)
    
    #get new max for cases and deaths
    ncmax, ndmax = get_country_max(Data)
    
    #scales data down to [0,1] interval on both x,y axis
    Data = scale_data(Data,ncmax,ndmax)
    
    #gets max values in each column of countries data matrix
    cmax,dmax = get_country_max(Data)
    
    #create earcons
    Earcons = create_earcons(cmax,dmax,rcsize)
    
    #create audio
    modamp = 2
    Audio = create_audio(Data,Earcons,Chords,modamp)
    
    #scale up  data/earcons for graph
    GData = scale_data(Data,ncmax,ndmax,False)
    GEarcons = scale_data(Earcons,ncmax,ndmax,False)
    make_data_earcon_plot(GData,GEarcons,countrycode,DesiredAudioFrame)
    
    
    #make population % graph earcon plot
    PData = scale_data(GData,popsize,popsize)
    PEarcons = scale_data(GEarcons,popsize,popsize)
    make_data_earcon_population_plot(PData,PEarcons,countrycode,DesiredAudioFrame)
    
    return Audio

if __name__ == '__main__': 
    
    #covid data from january 20th to july 10th 2020
    UScode = "US"
    USPop = 329064917

    #interpolate 
    fs = 44100
    seconds = 1
    DesiredAudioFrame = np.linspace(0, seconds, int(fs*seconds))

    #creates chords to be used for amplitude modulation
    Types = ["MajorW","Major7","Major","Minor7","Dim","Aug","Minor","FullyDim","Tri"]
    freq = 440
    Chords = make_chord_arrays(seconds,Types,freq)
    rcsize = int(np.sqrt(len(Types)))
    
    Audio = do_covid_calc(UScode,DesiredAudioFrame,Chords,rcsize,USPop)
    
