import numpy as np
import matplotlib.pyplot as plt
import sys
from CubicSpline import *

xs = np.array([0, 1.89, 2.88, 3.78, 4.64, 5.47, 6.29, 7.1, 7.92, 8.75, 9.58])
ys = np.arange(11)*10
t = np.linspace(0, 9.58, int(44100*9.58)) # How to sample
[position, velocity, acceleration] = sample_cubic_spline(xs, ys, t)

fs = 44100

v = velocity

vscale = 2**v
vscaled = np.reshape(vscale,(1,len(vscale)))
vscaledd = np.cumsum(vscaled, axis=0)
pscaled = np.cumsum(vscaledd, axis=1)
pnew = pscaled.flatten()
vnew = vscaledd.flatten()


plt.figure(figsize=(6, 18))
plt.subplot(311)
plt.plot(t, pnew)
plt.xlabel("Time (Sec)")
plt.ylabel("Position (Meters)")
plt.title("Position")

plt.subplot(312)
plt.plot(t, vnew)
plt.xlabel("Time (Sec)")
plt.ylabel("Velocity (Meters/Sec)")
plt.title("Velocity")

plt.subplot(313)
plt.plot(t, acceleration)
plt.xlabel("Time (Sec)")
plt.ylabel("Acceleration (Meters/Sec^2)")
plt.title("Acceleration")

plt.tight_layout()
plt.show()