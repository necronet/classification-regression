import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    means = np.array([])

    for i in np.unique(y):
        class_mean = np.mean( X[np.where(y==i)[0]],axis=0)
        means = np.append(means, class_mean)

    covmat = np.cov(X, rowvar=False)

    return means.reshape(5,2),covmat

def qdaLearn(X,y):

    # Like LDA, the QDA classifier results from assuming that the
    # observations from each class are drawn from a Gaussian distribution, and
    # plugging estimates for the parameters into Bayes’ theorem in order to perform
    # prediction.However, unlike LDA, QDA assumes that each class has its own
    # covariance matrix. (Bishop 2006)

    means = np.array([])
    covmats = np.array([])

    for i in np.unique(y):
        class_mean = np.mean( X[np.where(y==i)[0]],axis=0)
        covmat = np.cov(X[np.where(y==i)[0]], rowvar=False)
        means = np.append(means, class_mean)
        covmats = np.append(covmats, covmat)

    return means.reshape(5,2),covmats.reshape(5,4)

def ldaTest(means,covmat,Xtest,ytest):
    cov_det = np.linalg.det(covmat)
    inverse_cov = np.linalg.inv(covmat)
    p = Xtest.shape[1]

    pdf= np.zeros((Xtest.shape[0],means.shape[0]))

    for i in range(means.shape[0]):
        denominator = pow(pi*2,p/2)*sqrt(cov_det)
        for j in range(Xtest.shape[0]):
            A = Xtest[j,:] - means[i]
            S = np.dot(A, np.dot(inverse_cov,A.T) )
            pdf[j,i] = np.exp(-0.5*S) /denominator

    ypred = np.argmax(pdf,1)
    ytest = ytest.reshape(ytest.size)
    ypred = ypred + 1
    acc = 100*np.mean((ypred == ytest))

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):

    p = Xtest.shape[1]
    pdf= np.zeros((Xtest.shape[0],means.shape[0]))

    for i in range(means.shape[0]):

        covmat = covmats[i].reshape(2,2)
        cov_det = np.linalg.det(covmat)
        inverse_cov = np.linalg.inv(covmat)

        denominator = pow(pi*2,p/2)*sqrt(cov_det)

        for j in range(Xtest.shape[0]):
            A = Xtest[j,:] - means[i]
            S = np.dot(A, np.dot(inverse_cov,A.T) )
            pdf[j,i] = np.exp(-0.5*S) /denominator

    ypred = np.argmax(pdf,1)
    ytest = ytest.reshape(ytest.size)
    ypred = ypred + 1
    acc = 100*np.mean((ypred == ytest))

    return acc,ypred
