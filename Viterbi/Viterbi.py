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

def calcA(S):#transistion matrix A (stores distance between elements in S)
    LS = int(len(S))
    A=[0.0]*(LS*LS)
    A = np.reshape(A, (LS, LS))
    for i in range(LS):
        for j in range(LS):
            A[i,j] = np.sqrt((S[j,0]-S[i,0])**2 + (S[j,1] - S[i,1])**2)
    return A

def calcB(Y,S):#emission matrix B (stores distance from each point in Y to each point in S)
    LY = int(len(Y))
    LS = int(len(S))
    B=[0.0]*(LY*LS)
    B = np.reshape(B, (LS, LY))
    for i in range(LS):
        for j in range(LY):
            B[i,j] = np.sqrt((Y[j,0]-S[i,0])**2 + (Y[j,1] - S[i,1])**2)
    return B

def calcP(S,K):
    P = np.zeros(S.shape[0])
    for i in range(0,K-1):
        P[i] = 1000
    P[np.random.randint(K-1)] = 0
    return P

def calcT(A,B,S,Y,K,N,T1,T2,P):
    
    for i in range(K):
        T1[i,0] = P[i] + B[i,0]
        T2[i,0] = i
    for j in range(1,N):
        for i in range(K):
            T1[i,j] = np.min(T1[:,j-1]+A[:,i]+B[i,j])
            T2[i,j] = np.argmin(T1[:,j-1]+A[:,i]+B[i,j])
    
    '''
    X = np.zeros(N, dtype=int)
    X[N-1] = T2[K-1,N-1]
    a = X[N-1]
    for i in range(N-1,-1,-1):
        X[i] = T2[a,i]
        a = X[i]   
    '''
    
    X = np.zeros(57, dtype=int)
    x = len(X)
    X[x-1] = T2[x-1,N-1]
    a = X[x-1]
    for i in range(x-1,-1,-1):
        X[i] = T2[a,i]
        a = X[i]
    
    return X

def DoViterbi(S,Y):
    K = len(S)
    N = len(Y)
    T1 = np.ndarray(shape=(K,N), dtype=float)
    T2 = np.ndarray(shape=(K,N), dtype=float)
    A = calcA(S)
    B = calcB(Y,S)
    P = np.zeros(S.shape[0])
    #P = calcP(S,K)
    X = calcT(A,B,S,Y,K,N,T1,T2,P)
    G = S[X, :]
    return G

if __name__ == '__main__':
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
    
    #G = DoViterbi(S,Y)
    #plt.scatter(G[:, 0], G[:, 1], c = np.arange(G.shape[0]))

"""
start the backtracing from a state that doesn't have the min cost
start off with large values in Pi except for one element, to encourage the path to start off at that element.  
compare the results
"""

K = len(S)
N = len(Y)
T1 = np.ndarray(shape=(K,N), dtype=float)
T2 = np.ndarray(shape=(K,N), dtype=float)
A = calcA(S)
B = calcB(Y,S)
P = np.zeros(S.shape[0])
#P = calcP(S,K)
X = calcT(A,B,S,Y,K,N,T1,T2,P)
G = S[X, :]
plt.scatter(G[:, 0], G[:, 1], c = np.arange(G.shape[0]))
