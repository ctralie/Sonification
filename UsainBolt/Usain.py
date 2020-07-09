import numpy as np
import matplotlib.pyplot as plt

def create_chords(freqscale, freq, arr, acceleration, arrl, typeofmod=0):
    
    
    ##typeofmod: position arr = 0 velocity arr = 1
    
    
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

    #making sin waves
    if typeofmod == 0:
        AO = np.sin(2*np.pi*NO/freqscale)
        AT = np.sin(2*np.pi*NT/freqscale)
        ATH = np.sin(2*np.pi*NTH/freqscale)
        AF = np.sin(2*np.pi*NF/freqscale)
        #return chord
        return AO + AT + ATH + AF
    else:
        AO = np.sin(2*np.pi*NO*freqscale)
        AT = np.sin(2*np.pi*NT*freqscale)
        ATH = np.sin(2*np.pi*NTH*freqscale)
        AF = np.sin(2*np.pi*NF*freqscale)
        #return chord
        return AO + AT + ATH + AF
        