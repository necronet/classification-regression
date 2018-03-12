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
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD

    means = np.array([])

    for i in np.unique(y):
        class_mean = np.mean( X[np.where(y==i)[0]],axis=0)
        means = np.append(means, class_mean)

    covmat = np.cov(X, rowvar=False)

    return means.reshape(5,2),covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # Like LDA, the QDA classifier results from assuming that the
    # observations from each class are drawn from a Gaussian distribution, and
    # plugging estimates for the parameters into Bayes’ theorem in order to perform
    # prediction.However, unlike LDA, QDA assumes that each class has its own
    # covariance matrix. (Bishop 2006)
    # IMPLEMENT THIS METHOD

    means = np.array([])
    covmats = np.array([])

    for i in np.unique(y):
        class_mean = np.mean( X[np.where(y==i)[0]],axis=0)
        covmat = np.cov(X[np.where(y==i)[0]], rowvar=False)
        means = np.append(means, class_mean)
        covmats = np.append(covmats, covmat)

    return means.reshape(5,2),covmats.reshape(5,4)

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD



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
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD


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

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')


# # LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))
#
# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()
#
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))

plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)

colors = ("red", "green", "blue","black","orange")
for i in np.unique(ytest):
    index = i.astype(int)-1
    ys = np.where(ytest==i)[0]
    plt.scatter(Xtest[ys,0],Xtest[ys,1], c=colors[index])

plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)

for i in np.unique(ytest):
    index = i.astype(int)-1
    ys = np.where(ytest==i)[0]
    plt.scatter(Xtest[ys,0],Xtest[ys,1], c=colors[index])

plt.title('QDA')

plt.show()
