#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:44:22 2020

@author: michaeltecce
"""

from SyntheticCurves import *
import numpy as np
import matplotlib.pyplot as plt

'''
state space S = space of all audio grain (blue dots)
(pi) array = array of initial costs
Y = observations array (ribbon)
'''

def getCSM(X, Y):
    """
    Return the cross-similarity matrix between two point-clouds;
    that is, a matrix whose ijth entry is the Euclidean distance
    between Xi and Yj
    Parameters
    ----------
    X: ndarray(M, d)
        First point cloud with M points in d dimensions
    Y: ndarray(N, d)
        Section point cloud with N points in d dimensions
    Returns
    -------
    C: ndarray(M, N)
        All pairs of distances from one to the next
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def calcP(S,K):
    P = np.zeros(S.shape[0])
    for i in range(0,K-1):
        P[i] = 1000
    P[np.random.randint(K-1)] = 0
    return P

def do_viterbi(S,Y,P):
    """
    Perform the Viterbi algorithm
    Parameters
    ---------
    S: ndarray(K, d)
        The possible states
    Y: ndarray(N, d)
        The observations
    P: ndarray(K)
        The starting state costs
    Returns
    -------
    X: ndarray(N, dtype=int)
        Indices of an optimal sequence of states
    """
    K = len(S)
    N = len(Y)
    T1 = np.ndarray(shape=(K,N), dtype=float)
    T2 = np.ndarray(shape=(K,N), dtype=float)
    A = getCSM(S, S)
    B = getCSM(S, Y)

    for i in range(K):
        T1[i,0] = P[i] + B[i,0]
        T2[i,0] = i
    for j in range(1,N):
        for i in range(K):
            T1[i,j] = np.min(T1[:,j-1]+A[:,i]+B[i,j])
            T2[i,j] = np.argmin(T1[:,j-1]+A[:,i]+B[i,j])
    
    X = np.zeros(N, dtype=int)
    X[N-1] = T2[K-1,N-1]
    a = X[N-1]
    for i in range(N-1,-1,-1):
        X[i] = T2[a,i]
        a = X[i]   
    
    return X

def do_2D_synthetic_test():
    """
    Do a test of Viterbi with a figure 8 with
    states in a square sampled around it
    """
    t = np.linspace(0, 1, 100)
    Y = get2DFigure8(t)
    np.random.seed(0)
    S = np.random.rand(1000, 2)
    S = S*3
    S[:, 0] -= 1.5
    S[:, 1] -= 1.5
    
    plt.scatter(S[:, 0], S[:, 1])
    plt.scatter(Y[:, 0], Y[:, 1], c = t)
    plt.axis('equal')
    plt.show()

    P = np.zeros(S.shape[0])
    #P = calcP(S,K)
    X = do_viterbi(S, Y, P)
    G = S[X, :]
    plt.scatter(G[:, 0], G[:, 1], c = np.arange(G.shape[0]))
    plt.show()

if __name__ == '__main__':
    do_2D_synthetic_test()
