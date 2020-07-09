import numpy as np
import matplotlib.pyplot as plt

def create_chords(freqscale, freq, arr, acceleration, arrl, typeofmod=0):
    
    """
    Creates an audio array comprised of major and minor 7th chords from a time-series array
    Parameters
    ----------
    freqscale:   int
        value to scale the frequency range by
    freq:    int
        base frequency for start of array chord
    arr:    array()
        time-series to be sonified
    acceleration:   array()
        a derivative(or 2nd derivative) of arr
        determines if chord will be major or minor
    arrl:   int
        last value of np.linspace array which holds the time series
    typeofmod:   int
        indicator for what type of array was sent in for sonification
        0 = position array, 1 = velocity array
        
    Return
    ------
    CHORD:   array()
        array comprised of 4 distinct sin waves from the arr param
    
    """
    fs = 44100
    
    #setting 1st and 3rd harmonies
    NO = arr[:] * freq * (2**(0/12))
    NTH = arr[:] * freq * (2**(7/12))
    
    #creating array for 2nd and 4th harmonies 
    NT = np.linspace(0, arrl, int(fs * arrl))
    NF = np.linspace(0, arrl, int(fs * arrl))

    for i in range(len(arr)):
        if acceleration[i] >= 0:
            #do major 7
            NT[i] = arr[i] * freq * (2**(4/12))
            NF[i] = arr[i] * freq * (2**(11/12))
        else:
            #do minor 7th
            NT[i] = arr[i] * freq * (2**(3/12))
            NF[i] = arr[i] * freq * (2**(10/12))
    
    #chose scale type
    if typeofmod == 0:
        freqscale = 1/freqscale
    
    #making sin waves
    AO = np.sin(2*np.pi*NO*freqscale)
    AT = np.sin(2*np.pi*NT*freqscale)
    ATH = np.sin(2*np.pi*NTH*freqscale)
    AF = np.sin(2*np.pi*NF*freqscale)
    
    #return chord
    CHORD =  AO + AT + ATH + AF
    return CHORD

        