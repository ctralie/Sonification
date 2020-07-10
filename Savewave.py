import scipy.io.wavfile as wavfile # We will use this library to load in audio
import numpy as np

def save_wavfile(filename, fs, x):
    
    """
    @author: ctralie
    
    Save audio to a wave file
    Parameters
    ----------
    filename: string
        Name of the file you want to save to
    fs: int
        Sample rate
    x: ndarray(N)
        Numpy array of audio samples
    """
    y = x-np.mean(x)
    y = y/np.max(np.abs(y))
    y = np.array(y*32768, dtype=np.int16)
    print(np.min(y))
    print(np.max(y))
    wavfile.write(filename, fs, y)