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
