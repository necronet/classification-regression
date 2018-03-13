import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
from discriminant import ldaLearn, qdaLearn, ldaTest, qdaTest
from linear_regression import learnOLERegression, testOLERegression, learnRidgeRegression

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    

    # IMPLEMENT THIS METHOD
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))

    # IMPLEMENT THIS METHOD
    return Xp

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

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
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

# plt.show()
# # Problem 4
# k = 101
# lambdas = np.linspace(0, 1, num=k)
# i = 0
# mses4_train = np.zeros((k,1))
# mses4 = np.zeros((k,1))
# opts = {'maxiter' : 20}    # Preferred value.
# w_init = np.ones((X_i.shape[1],1))
# for lambd in lambdas:
#     args = (X_i, y, lambd)
#     w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
#     w_l = np.transpose(np.array(w_l.x))
#     w_l = np.reshape(w_l,[len(w_l),1])
#     mses4_train[i] = testOLERegression(w_l,X_i,y)
#     mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
#     i = i + 1
# fig = plt.figure(figsize=[12,6])
# plt.subplot(1, 2, 1)
# plt.plot(lambdas,mses4_train)
# plt.plot(lambdas,mses3_train)
# plt.title('MSE for Train Data')
# plt.legend(['Using scipy.minimize','Direct minimization'])
#
# plt.subplot(1, 2, 2)
# plt.plot(lambdas,mses4)
# plt.plot(lambdas,mses3)
# plt.title('MSE for Test Data')
# plt.legend(['Using scipy.minimize','Direct minimization'])
# plt.show()
#
#
# # Problem 5
# pmax = 7
# lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
# mses5_train = np.zeros((pmax,2))
# mses5 = np.zeros((pmax,2))
# for p in range(pmax):
#     Xd = mapNonLinear(X[:,2],p)
#     Xdtest = mapNonLinear(Xtest[:,2],p)
#     w_d1 = learnRidgeRegression(Xd,y,0)
#     mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
#     mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
#     w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
#     mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
#     mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
#
# fig = plt.figure(figsize=[12,6])
# plt.subplot(1, 2, 1)
# plt.plot(range(pmax),mses5_train)
# plt.title('MSE for Train Data')
# plt.legend(('No Regularization','Regularization'))
# plt.subplot(1, 2, 2)
# plt.plot(range(pmax),mses5)
# plt.title('MSE for Test Data')
# plt.legend(('No Regularization','Regularization'))
# plt.show()
