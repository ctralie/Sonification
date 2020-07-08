import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
sys.path.append('..')
from CurvatureTools import *

def sample_cubic_spline(xs, ys, t):
    """
    Create a 1D cubic spline and sample it
    Parameters
    ----------
    xs: ndarray(M)
        The x coordinates of the points
    ys: ndarray(M)
        The y coordinates of the points
    t: ndarray(N)
        The x locations at which to sample the points.
        It will be clamped to the interval of the time series
    Returns
    -------
    spline: ndarray(N)
        The y coordinates of the sampled spline
    """
    n = len(ys)-1
    a = np.array(ys)
    b = np.zeros(n)
    d = np.zeros(n)
    h = xs[1::]-xs[0:-1]
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = 3*(a[i+1]-a[i])/h[i] - 3*(a[i]-a[i-1])/h[i-1]
    c = np.zeros(n+1)
    l = np.zeros(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)
    l[0] = 1
    for i in range(1, n):
        l[i] = 2*(xs[i+1]-xs[i-1])-h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i]
    l[n] = 1
    for j in range(n-1, -1, -1):
        c[j] = z[j]-mu[j]*c[j+1]
        b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    s1 = np.array([])
    s2 = np.array([])
    s3 = np.array([])
    for i in range(n):
        if i == n-1:
            ti = t[(t >= xs[i])*(t <= xs[i+1])]
        else:
            ti = t[(t >= xs[i])*(t < xs[i+1])]
        # Direct spline function
        f = a[i] + b[i]*(ti-xs[i]) + c[i]*(ti-xs[i])**2 + d[i]*(ti-xs[i])**3
        s1 = np.concatenate((s1, f))
        # Derivative function
        f = b[i] + 2*c[i]*(ti-xs[i]) + 3*d[i]*(ti-xs[i])**2
        s2 = np.concatenate((s2, f))
        # Second derivative function
        f = 2*c[i] + 6*d[i]*(ti-xs[i])
        s3 = np.concatenate((s3, f))
    return [s1, s2, s3]