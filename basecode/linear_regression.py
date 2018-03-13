import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def learnOLERegression(X,y):
    w = np.dot(np.dot( (np.linalg.inv(np.dot(X.T,X))),  X.T),y)
    return w

def testOLERegression(w,Xtest,ytest):
    N = ytest.shape[0]
    mse = np.sum( np.square( ytest.T - (np.dot(w.T,Xtest.T)) ) ) / N
    return mse

def learnRidgeRegression(X,y,lambd):
    rigde_term = lambd * np.identity(X.shape[1])
    w = np.dot( np.dot( np.linalg.inv(np.dot( X.T, X ) + rigde_term ), X.T), y )
    return w


def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))
    N = x.shape[0]
    Xp = np.ones((N,p+1))
    for i in range(1,p+1):
        Xp[:,i] = np.power(x,i)

    return Xp

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    m = X.shape[0]
    RSS = y - (np.dot(w.T,X.T)).T
    RSS = np.sum( np.dot(RSS,RSS.T) )
    error = RSS + lambd*np.sum(np.dot(w,w.T))/m
    error_grad = (np.dot(w.T,np.dot(X.T,X)) - np.dot(y.T,X)) / m + 2*lambd*w.T
    error_grad = np.ndarray.flatten(error_grad)

    return error, error_grad
