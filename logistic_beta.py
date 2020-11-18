# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:21:46 2020

@author: Akshay Bhatia
"""

import numpy as np
import sympy as sym
from scipy.special import expit

#from scipy.optimize import fsolve

W = np.array([94,130,125,88,79,53,40,54,51,52,22,25,44,47,42,26,38,36,9,25,18,20,23,19,13,17,15,17,21,26,17,22,24,21,31,20,17,14,15,13,21,19])
Z = np.array([299,261,197,160,115,98,110,103,97,61,73,88,92,82,68,73,63,38,46,44,47,46,34,31,30,31,42,43,49,47,53,52,47,60,43,33,26,30,31,39,35,36])

W = W.astype(np.float64)
Z = Z.astype(np.float64)

X = np.random.rand(6,20) #We have to use actual covariate values instead of this random uniform matrix
#X[1:] = 1.0
tau = 6.10   #q = 0.45, lambda = 1.5

print(X[5:])

def eta(X,beta,j):
    p = 0
    for i in range(6):
        p = p + X[i,j]*beta[i]
        p = expit(p)
    return 1.0/(1.0 + np.exp(-1*p))

def aux(X,vec,j):
    p = 0
    for i in range(6):
        p = p + X[i,j]*vec[i]
        p = expit(p)
    return np.exp(p)


u = np.zeros((20,1))
D = np.zeros((20,6))

    
beta = [1.0,1.0,1.0,1.0,1.0,1.0]  #initial guess
itern = 0

while True:
    for i in range(20):
        for j in range(6):
            D[i,j] = -1*X[j,i]*(eta(X,beta,i)**2)

    
    u[19] = -1*Z[19]/eta(X,beta,19)

    for i in range(19):
        u[i] = -1*Z[i]/eta(X,beta,i+1)    
    
    v = np.transpose(u)
    A = np.matmul(u,v)
    N = sym.Matrix(A)
    if(N.det()==0):
        A = A + np.eye(A.shape[0])*1e-7
    A_inv = np.linalg.inv(A)
    Y1 = np.matmul(A_inv,u) 
    Y2 = np.matmul(D,beta)
    Y2 = Y2.reshape((20, 1))
    Y = Y1 + Y2

    T1 = np.matmul(np.transpose(D),A)
    T2 = np.matmul(T1,D)
    M = sym.Matrix(T2)
    if(M.det()==0):
        T2 = T2 + np.eye(T2.shape[0])*1e-5
    T3 = np.linalg.inv(T2)
    T4 = np.matmul(T3,T1)
    
    beta1 = beta
    beta = np.matmul(T4,Y)
    res = beta - beta1

    itern += 1
    print(beta)
    tol = 5
    if ((np.abs(res) < tol).all()):
        print(beta)
        print(itern)
        print("Converged")
        break
    
    if itern == 1500:
        print("Did not converge")
        break

def likelihood(alpha):
    
    a = 0
    for j in range(20):
        p = 0
        for i in range(6):
            p += X[i,j]*beta[i]
            p = W[j]*p
            a += p
    
    b = 0
    for j in range(1,20):
        p = 0
        for i in range(6):
            p += X[i,j]*beta[i]
            p = expit(p)
            p = Z[j-1]*np.log(1+np.exp(p))
            p = expit(p)
            b += p

    c = 0
    for j in range(1,20):
        c += Z[j]*np.log(aux(X,alpha,j)*Z[j-1]-W[j] + tau)
        
    d = 0
    for j in range(1,20):
        d += aux(X,alpha,j)*(Z[j-1] - W[j])
        
    l = a - b + c - d - 20*tau
    return l

U = np.zeros((1,6))
