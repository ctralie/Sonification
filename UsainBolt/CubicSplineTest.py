import numpy as np
import matplotlib.pyplot as plt
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
    spline = np.array([])
    for i in range(n):
        if i == n-1:
            ti = t[(t >= xs[i])*(t <= xs[i+1])]
        else:
            ti = t[(t >= xs[i])*(t < xs[i+1])]
        f = a[i] + b[i]*(ti-xs[i]) + c[i]*(ti-xs[i])**2 + d[i]*(ti-xs[i])**3
        spline = np.concatenate((spline, f))
    return spline

xs = np.array([0, 1.89, 2.88, 3.78, 4.64, 5.47, 6.29, 7.1, 7.92, 8.75, 9.58])
ys = np.arange(11)*10
t = np.linspace(0, 9.58, int(44100*9.58)) # How to sample
position = sample_cubic_spline(xs, ys, t)
velocity = getCurvVectors(position[:, None], 1, 1)[1]*t.size/xs.size
velocityV = velocity.flatten()

VMag = np.sqrt(np.sum(velocity**2, axis=1))

plt.figure(figsize=(6, 12))
plt.subplot(211)
plt.plot(t, position)
plt.xlabel("Time (Sec)")
plt.ylabel("Position (Meters)")
plt.title("Position")

plt.subplot(212)
plt.plot(t, velocity)
plt.xlabel("Time (Sec)")
plt.ylabel("Velocity (Meters/Sec)")
plt.title("Velocity")

plt.show()