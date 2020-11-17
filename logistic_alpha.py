# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:21:46 2020

@author: Akshay Bhatia
"""

import numpy as np
import sympy as sym
from scipy.special import expit

#from scipy.optimize import fsolve

W = np.array([0, 0, 253, 490, 714, 992, 1236, 1472, 1739, 1923, 2290, 2541, 2844, 3076, 3245, 3571, 3825, 4219, 4349, 4672])
Z = np.array([1, 486, 968, 1445, 1971, 2498, 2982, 3570, 3997, 4589, 5161, 5658, 6161, 6569, 7168, 7726, 8314, 8718, 9210, 9416])

W = W.astype(np.float64)
Z = Z.astype(np.float64)

X = np.random.rand(6,20)
X[1:] = 1.0
tau = 5.0

def eta(X,vec,j):
    p = 0
    for i in range(6):
        p = p + X[i,j]*vec[i]
        p = expit(p)
    return 1.0/(1.0 + np.exp(-1*p))

def aux(X,vec,j):
    p = 0
    for i in range(6):
        p = p + X[i,j]*vec[i]
        p = expit(p)
    return 1.0 + np.exp(p)


u = np.zeros((20,1))
D = np.zeros((20,6))

    
alpha = [1.0,1.0,1.0,1.0,1.0,1.0]
itern = 0

np.seterr(divide='ignore', invalid='ignore')

while True:
    for i in range(20):
        for j in range(6):
            D[i,j] = -1*X[j,i]*(eta(X,alpha,i)**2)
    
    for i in range(19):
        p = aux(X,alpha,i)
        r = Z[i] - W[i+1]
        u[i] = r*p*((1/(eta(X,alpha,i)*r*p) + tau) - 1.0)   
    u[19] = u[18]
    
    v = np.transpose(u)
    A = np.matmul(u,v)
    N = sym.Matrix(A)
    if(N.det()==0):
        A = A + np.eye(A.shape[0])
    A_inv = np.linalg.inv(A)
    Y1 = np.matmul(A_inv,u) 
    Y2 = np.matmul(D,alpha)
    Y2 = Y2.reshape((20, 1))
    Y = Y1 + Y2

    T1 = np.matmul(np.transpose(D),A)
    T2 = np.matmul(T1,D)
    M = sym.Matrix(T2)
    if(M.det()==0):
        for i in range(T2.shape[0]):
            for j in range(T2.shape[1]):
                T2[i,j] += i*0.1
        #T2 = T2 + np.eye(T2.shape[0])*2
    T3 = np.linalg.inv(T2)
    T4 = np.matmul(T3,T1)
    
    alpha1 = alpha
    alpha = np.matmul(T4,Y)
    res = alpha - alpha1

    itern += 1
    tol = 0.1
    print(alpha)
    print(alpha1)
    if ((np.abs(res) < tol).all()):
        print(alpha)
        print(itern)
        print("Converged")
        break
    
    if itern == 10:
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

    