"""
Programmer: Chris Tralie
Purpose: To create a collection of functions for making families of curves and applying
random rotations/translations/deformations/reparameterizations to existing curves
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys


###################################################
#           TOPC Utility Functions                #
###################################################

def getRandomRigidTransformation(dim, std, special = False):
    """
    Generate a random rotation matrix and translation
    Parameters
    ---------
    dim: int
        Dimension of the embedding
    std: float
        Standard deviation of coordinates in embedding (used to help
        place a translation)
    
    Returns
    -------
    R: ndarray(dim, dim)
        Rotation matrix
    T: ndarray(dim)
        Translation vector
    """
    #Make a random rotation matrix
    R = np.random.randn(dim, dim)
    R, S, V = np.linalg.svd(R)
    if special and np.linalg.det(R) < 0:
        idx = np.arange(R.shape[0])
        idx[0] = 1
        idx[1] = 0
        R = R[idx, :]
    T = 5*std*np.random.randn(dim)
    return R, T

def applyRandomRigidTransformation(X, special = False):
    """
    Randomly rigidly rotate and translate a time-ordered point cloud
    Parameters
    ---------
    X: ndarray(N, dim)
        Matrix representing a time-ordered point cloud
    special: boolean
        Whether to restrict to the special orthogonal group
        (no flips; determinant 1)
    :return Y: Nxd matrix representing transformed version of X
    """
    dim = X.shape[1]
    CM = np.mean(X, 0)
    X = X - CM
    R, T = getRandomRigidTransformation(dim, np.std(X))
    return CM[None, :] + np.dot(X, R) + T[None, :]

def smoothCurve(X, Fac):
    """
    Use splines to smooth the curve
    Parameters
    ----------
    X: ndarray(N, d)
        A matrix representing a time-ordered point cloud
    Fac: int
        Smoothing factor
    Returns
    -------
    Y: ndarray(N*Fac, d)
        An (NxFac)xd matrix of a smoothed, upsampled point cloud
    """
    NPoints = X.shape[0]
    dim = X.shape[1]
    idx = range(NPoints)
    idxx = np.linspace(0, NPoints, NPoints*Fac)
    Y = np.zeros((NPoints*Fac, dim))
    NPointsOut = 0
    for ii in range(dim):
        Y[:, ii] = interp.spline(idx, X[:, ii], idxx)
        #Smooth with box filter
        y = (0.5/Fac)*np.convolve(Y[:, ii], np.ones(Fac*2), mode='same')
        Y[0:len(y), ii] = y
        NPointsOut = len(y)
    Y = Y[0:NPointsOut-1, :]
    Y = Y[2*Fac:-2*Fac, :]
    return Y

###################################################
#               Curve Families                    #
###################################################

#Note: All function assume the parameterization is given
#in the interval [0, 1]

###########################################
##########  Arbitrary Dimensions   ########
###########################################
def makeRandomWalkCurve(res, NPoints, dim):
    """
    Make a random walk curve with "NPoints" in dimension "dim"
    Parameters
    ----------
    res: int
        An integer specifying the resolution of the random walk grid
    NPoints: int
        Number of points in the curve
    dim: int
        Dimension of the ambient Euclidean space of the curve
    Returns
    -------
    X: ndarray(NPoints, dim)
        The final point cloud
    """
    #Enumerate all neighbors in hypercube via base 3 counting between [-1, 0, 1]
    Neighbs = np.zeros((3**dim, dim))
    Neighbs[0, :] = -np.ones((1, dim))
    idx = 1
    for ii in range(1, 3**dim):
        N = np.copy(Neighbs[idx-1, :])
        N[0] += 1
        for kk in range(dim):
            if N[kk] > 1:
                N[kk] = -1
                N[kk+1] += 1
        Neighbs[idx, :] = N
        idx += 1
    #Exclude the neighbor that's in the same place
    Neighbs = Neighbs[np.sum(np.abs(Neighbs), 1) > 0, :]

    #Pick a random starting point
    X = np.zeros((NPoints, dim))
    X[0, :] = np.random.choice(res, dim)

    #Trace out a random path
    for ii in range(1, NPoints):
        prev = np.copy(X[ii-1, :])
        N = np.tile(prev, (Neighbs.shape[0], 1)) + Neighbs
        #Pick a random next point that is in bounds
        idx = np.sum(N > 0, 1) + np.sum(N < res, 1)
        N = N[idx == 2*dim, :]
        X[ii, :] = N[np.random.choice(N.shape[0], 1), :]
    return X


###########################################
##############   2D Curves   ##############
###########################################
def getLissajousCurve(a, b, pt, A = 1, B = 1, delta = 0):
    """
    Return the curve with
    x = Asin(at + delta)
    y = Bsin(bt)
    More info here
    https://mathworld.wolfram.com/LissajousCurve.html
    Parameters
    ----------
    a: int
        Radian frequency of first dimension
    b: int
        Radian frequency of second dimension
    pt: ndarray(N)
        The time parameters at which to sample these loops.  They
        will complete a loop on all intervals of length 1, and the
        principal loop occurs on [0, 1]
    A: int
        Amplitude of first sinusoid (default 1)
    B: int
        Amplitude of second sinusoid (default 2)
    delta: float
        Phase shift of first sinusoid (default 0)
    Returns
    -------
    X: ndarray(N, 2)
        The final sampled points
    """
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = A*np.sin(a*t + delta)
    X[:, 1] = B*np.sin(b*t)
    return X

def get2DFigure8(pt):
    """
    Return a figure 8 curve parameterized on [0, 1]
    Parameters
    ----------
    pt: ndarray(N)
        The time parameters at which to sample the figure 8.
        It completes a loop on all intervals of length 1, and the
        principal loop occurs on [0, 1]
    Returns
    -------
    X: ndarray(N, 2)
        Point cloud in 2 dimensions
    """
    return getLissajousCurve(1, 2, pt)


def getPinchedCircle(pt):
    """
    Return a pinched circle paramterized on [0, 1]
    Parameters
    ----------
    pt: ndarray(N)
        The time parameters at which to sample the figure 8.
        It completes a loop on all intervals of length 1, and the
        principal loop occurs on [0, 1]
    Returns
    -------
    X: ndarray(N, 2)
        Point cloud in 2 dimensions
    """
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    X[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    return X

def getEpicycloid(R, r, pt):
    """
    Return an epicycloid parameterized on [0, 1]
    More info here: https://en.wikipedia.org/wiki/Epicycloid
    Parameters
    ----------
    R: float
        Outer radius/frequency
    r: float
        Inner radius/frequency
    pt: ndarray(N)
        The time parameters at which to sample this curve
    Returns
    -------
    ndarray(N, 2)
        The sampled point cloud
    """
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = (R+r)*np.cos(t) - r*np.cos(t*(R+r)/r)
    X[:, 1] = (R+r)*np.sin(t) - r*np.sin(t*(R+r)/r)
    return X

def getTschirnhausenCubic(a, pt):
    """
    Return the plane curve defined by the polar equation
    r = asec^3(theta/3), which makes a ribbon shape
    """
    N = len(pt)
    t = 5*(pt-0.5)
    X = np.zeros((N, 2))
    X[:, 0] = a*(1-3*t**2)
    X[:, 1] = a*t*(3-t**2)
    X = 2*X/np.max(np.abs(X))
    return X


###########################################
##############   3D Curves   ##############
###########################################

def getVivianiFigure8(a, pt):
    """
    Return the curve that results from the intersection of
    a sphere of radius 2a centered at the origin and a cylinder
    centered at (a, 0, 0) of radius a (the figure 8 I have is
    a 2D projection of this)
    Parameters
    ----------
    a: float
        Radius of the sphere
    pt: ndarray(N)
        The time parameters at which to sample this curve
    Returns
    -------
    X: ndarray(N, 3)
        The points sampled on the curve
    """
    N = len(pt)
    t = 4*np.pi*pt - np.pi
    X = np.zeros((N, 3))
    X[:, 0] = a*(1+np.cos(t))
    X[:, 1] = a*np.sin(t)
    X[:, 2] = 2*a*np.sin(t/2)
    return X


def getTorusKnot(p, q, pt):
    """
    Return a p-q torus not parameterized on [0, 1]
    Parameters
    ----------
    p: int
        Radian frequency around large torus loop
    q: int
        Radian frequency around small torus loop
    pt: ndarray(N)
        The time parameters at which to sample this curve
    Returns
    -------
    X: ndarray(N, 3)
        Points sampled on the curve
    """
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 3))
    r = np.cos(q*t) + 2
    X[:, 0] = r*np.cos(p*t)
    X[:, 1] = r*np.sin(p*t)
    X[:, 2] = -np.sin(q*t)
    return X

def getConeHelix(c, NPeriods, pt):
    """
    Return a helix wrapped around a double ended cone
    Parameters
    ----------
    c: float
        Half the maximum radius of the cone on the interval [0, 1]
    NPeriods: int
        Number of times to wrap around the cones
    pt: ndarray(N)
        The time parameters at which to sample this curve
    Returns
    -------
    X: ndarray(N, 3)
        Points sampled on the curve
    """
    N = len(pt)
    t = NPeriods*2*np.pi*pt
    zt = c*(pt-0.5)
    r = zt
    X = np.zeros((N, 3))
    X[:, 0] = r*np.cos(t)
    X[:, 1] = r*np.sin(t)
    X[:, 2] = zt
    return X


def get_lims(X, dim, pad=0.1):
    """
    Return the limits around a dimension with some padding
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud in d dimensions
    dim: int
        Dimension to extract limits from
    pad: float
        Factor by which to pad
    """
    xlims = [np.min(X[:, dim]), np.max(X[:, dim])]
    xlims[0] = xlims[0]-(xlims[1]-xlims[0])*pad
    xlims[1] = xlims[1]+(xlims[1]-xlims[0])*pad
    return xlims

#A class for doing animation of the sliding window
class CurveAnimator(animation.FuncAnimation):
    """
    Create a video of a sampled curve evolving over time
    """
    def __init__(self, X, filename, fps=30, figsize=(8, 8), 
                title = 'Evolving Curve Animation', bitrate=10000, cmap='magma_r'):
        """
        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud
        filename: string
            Output name of video
        fps: int
            Frames per second of the video
        figsize: tuple(2)
            Width x height of figure
        title: string
            Title of the video
        bitrate: int
            Output bitrate of the video
        cmap: string
            The colormap to use
        """
        assert(X.shape[1] >= 2)
        self.fig = plt.figure(figsize=figsize)
        self.X = X
        self.bgcolor = (0.2, 0.2, 0.2)
        t = np.linspace(0, 1, X.shape[0])
        c = plt.get_cmap(cmap)
        C = c(np.array(np.round(255*t), dtype=np.int32))
        self.C = C[:, 0:3]
        self.xlims = get_lims(X, 0, 0.2)
        self.ylims = get_lims(X, 1, 0.2)
        self.filename = filename

        ax = None
        if X.shape[1] > 2:
            ax = self.fig.add_subplot(111, projection='3d')
            ax.set_xlim(get_lims(X, 0))
            ax.set_ylim(get_lims(X, 1))
            ax.set_zlim(get_lims(X, 2))
        else:
            ax = self.fig.add_subplot(111)
            ax.set_xlim(get_lims(X, 0))
            ax.set_ylim(get_lims(X, 1))
            ax.set_facecolor(self.bgcolor)
        ax.set_facecolor(self.bgcolor)
        ax.set_xlabel("Column 1")
        ax.set_ylabel("Column 2")
        if X.shape[1] > 2:
            ax.set_zlabel("Column 3")
        self.ax = ax

        #Setup animation thread
        self.n_frames = X.shape[0]
        animation.FuncAnimation.__init__(self, self.fig, func = self._draw_frame, frames = self.n_frames, interval = 10)

        #Write movie
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title=title, comment='Evolving curve animation')
        writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate = bitrate)
        self.save(filename, writer = writer)

    def _draw_frame(self, i):
        print("Rendering frame {} of {} of {}".format(i+1, self.n_frames, self.filename))
        X = self.X
        c = self.C[i, :]
        c = c[None, :]
        if X.shape[1] == 2:
            self.ax.scatter([X[i, 0]], [X[i, 1]], c=c)
        elif X.shape[1] >= 3:
            self.ax.scatter([X[i, 0]], [X[i, 1]], [X[i, 2]], c=c)
        self.ax.set_title("Frame {}".format(i+1))

if __name__ == '__main__':
    t = np.linspace(0, 1, 180)
    X = getVivianiFigure8(2, t)
    CurveAnimator(X, "viviani.mp4", fps=30, title='Viviani Figure 8')

    t = np.linspace(0, 1, 100)
    X = getEpicycloid(3, 1, t)
    CurveAnimator(X, "epicycloid.mp4", fps=10, title='3-1 Epicycloid')