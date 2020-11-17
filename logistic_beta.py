# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:21:46 2020

@author: Akshay Bhatia
"""

import numpy as np
import sympy as sym

#from scipy.optimize import fsolve

W = np.array([0, 0, 253, 490, 714, 992, 1236, 1472, 1739, 1923, 2290, 2541, 2844, 3076, 3245, 3571, 3825, 4219, 4349, 4672])
Z = np.array([1, 486, 968, 1445, 1971, 2498, 2982, 3570, 3997, 4589, 5161, 5658, 6161, 6569, 7168, 7726, 8314, 8718, 9210, 9416])

W = W.astype(np.float64)
Z = Z.astype(np.float64)

X = np.random.rand(6,20) #We have to use actual covariate values instead of this random uniform matrix
X[1:] = 1.0

def eta(X,beta,j):
    p = 0
    for i in range(6):
        p = p + X[i,j]*beta[i]
    return 1.0/(1.0 + np.exp(-1*p))
u = np.zeros((20,1))
D = np.zeros((20,6))

    
beta = [1.0,1.0,1.0,1.0,1.0,1.0]  #initial guess
itern = 0

while True:
    for i in range(20):
        for j in range(6):
            D[i,j] = -1*X[j,i]*(eta(X,beta,i)**2)

    '''
    for i in range(20):
        u[i] = -1*Z[i]/eta(X,beta,i)
    '''
    
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
    tol = 0.01
    if ((np.abs(res) < tol).all()):
        print(beta)
        print(itern)
        print("Converged")
        break
    
    if itern == 1500:
        print("Did not converge")
        break
    #print("iteration starts")
    #print(beta)
    #print(beta1)
    #print(res)
    #print("iteration ends")

 
'''
u[19] = -1*Z[18]/eta(X,beta,19)

for i in range(19):
    u[i] = -1*Z[i]/eta(X,beta,i+1)
'''

    