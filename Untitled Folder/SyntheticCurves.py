"""
Programmer: Chris Tralie
Purpose: To create a collection of functions for making families of curves and applying
random rotations/translations/deformations/reparameterizations to existing curves
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
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
    :param X: Nxd matrix representing a time-ordered point cloud
    :param Fac: Smoothing factor
    :return Y: An (NxFac)xd matrix of a smoothed, upsampled point cloud
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

def makeRandomWalkCurve(res, NPoints, dim):
    """
    Make a random walk curve with "NPoints" in dimension "dim"
    :param res: An integer specifying the resolution of the random walk grid
    :param NPoints: Number of points in the curve
    :param dim: Dimension of the ambient Euclidean space of the curve
    :return X
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

###################################################
#               Curve Families                    #
###################################################

#Note: All function assume the parameterization is given
#in the interval [0, 1]


#######2D Curves
def getLissajousCurve(A, B, a, b, delta, pt):
    """
    Return the curve with
    x = Asin(at + delta)
    y = Bsin(bt)
    """
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = A*np.sin(a*t + delta)
    X[:, 1] = B*np.sin(b*t)
    return X

def get2DFigure8(pt):
    """Return a figure 8 curve parameterized on [0, 1]"""
    return getLissajousCurve(1, 1, 1, 2, 0, pt)


def getPinchedCircle(pt):
    """Return a pinched circle paramterized on [0, 1]"""
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    X[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    return X

def getEpicycloid(R, r, pt):
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = (R+r)*np.cos(t) - r*np.cos(t*(R+r)/r)
    X[:, 1] = (R+r)*np.sin(t) - r*np.sin(t*(R+r)/r)
    return X

def getTschirnhausenCubic(a, pt):
    """
    Return the plane curve defined by the polar equation
    r = asec^3(theta/3)
    """
    N = len(pt)
    t = 5*(pt-0.5)
    X = np.zeros((N, 2))
    X[:, 0] = a*(1-3*t**2)
    X[:, 1] = a*t*(3-t**2)
    X = 2*X/np.max(np.abs(X))
    return X




