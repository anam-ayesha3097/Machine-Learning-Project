#!/usr/bin/env python
# coding: utf-8

# In[97]:


#Importing all the required Libraries to perform L2-regularized Regression
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import KFold
from statistics import mean


# In[98]:


#Import the loadData() and normalizeData() function from utils file
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)


# In[99]:


#Seperate the dataset and target in train(100) and test datapoints
#In this experiment use on the 3rd feature of AutMPG dataset i.e. HorsePower
x_train = X_n[0:100,2]
x_test = X_n[100:,2]

y_train = t[0:100]
y_test = t[100:]

x_train = np.reshape(x_train, (-1, 1))
x_test = np.reshape(x_test, (-1, 1))


# In[100]:


#MSE(Mean Squared Error) Loss function
def getLoss(y_hyp, y_target):
    #MSE - Mean((target - hypothesis)**2)
    mse = np.square(np.subtract(y_target, y_hyp)).mean()
    return mse


# In[101]:


#Calculating Ridge Regression based on Normalizing the Regularizer with Lambda(lr)
#E(w) = (Xw- t).T * (Xw -t) + lambda * norm(weights**2)
# t -> target
# X -> datapoints
# w -> weights
def getLossReg(y_hyp, y_target, weights, lr):
    error = np.dot(np.subtract(y_hyp, y_target).T, np.subtract(y_hyp, y_target))
    L2_Weight = np.dot(weights.T, weights)
    L2_norm = np.linalg.norm(L2_Weight)
    regularize = lr * L2_norm
    loss = np.add(error.sum(), regularize)
    return loss


# In[102]:


#calculating weights by applying the lambda(learning rate) in least square method
#w = inverse(lambda*I + X.T* X) X.T * t
# t -> target
# X -> datapoints
# w -> weights
def getWeights(data, target, lr):
    identityMatrix = np.identity(data.shape[1])
    regularize = lr * identityMatrix
    regularizeInversion = np.add(np.dot(data.T, data), regularize)
    inverted = np.linalg.inv(regularizeInversion)
    target = np.reshape(target, (-1, 1))
    weights = np.dot(np.dot(regularizeInversion, data.T), target)
    bias = np.subtract(target, np.dot(data, weights))
    return weights, bias


# In[103]:


def trainL2Regularization(data, target, lr, degree):
    
    #Passing data through the basis function
    phiX = degexpand(data, degree)
    
    #Calculating the weights and bias based on ridge regression formula
    weights, bias = getWeights(phiX, target, lr)
    
    #Calculating the model predicted value
    y_hyp = np.dot(phiX, weights)
    
    #Calculating the Loss based on MSE Loss or Regularized Least Square Loss Function
    #To check for Regularized least square loss just comment MSE and uncomment rls
    L2_mse = getLoss(y_hyp, target)
    #L2_rls = getLossReg(y_hyp, target, weights, lr)
    return y_hyp , L2_mse


# In[106]:


#Initialising Learning Rates
learning_rate = [0, 0.01, 0.1, 1, 10, 100, 1000]

#Implementing 10 Fold-Cross Validation to analyse the best lambda value for the given problem 
#shuffle = True will give different data points each time we train our model and thus will generate plots differently everytime
kfold = KFold(n_splits=10, shuffle=True)

trainP = []
valP = []
trainL = []
valL = []
valAvg = []

for lr in learning_rate:
    #Getting the Split train and validation index
    for train, val in kfold.split(x_train):
        #Segregating the train and validation datasets based on the k-fold cross-validation indexes
        xtrain_data = x_train[train]
        xtrain_data = np.reshape(train , (-1, 1))
        target = y_train[train]
        target = np.reshape(target, (-1, 1))
        #Getting the loss and prediction for train dataset
        predict, loss = trainL2Regularization(xtrain_data, target, lr, 8)
        trainP.append(predict)
        trainL.append(loss)
        xval_data = x_train[val]
        xval_data = np.reshape(val , (-1, 1))
        target_val = y_train[val]
        target_val = np.reshape(target_val , (-1, 1))
        #Getting the loss and prediction for validation dataset
        val_predict, val_loss = trainL2Regularization(xval_data, target_val, lr, 8)
        valP.append(val_predict)
        valL.append(val_loss)
    if(lr != 0):
        #Calculating average validation loss for each learning rates
        valAvg.append(np.average(valL))
        valL.clear()
        xtrain_data = np.zeros((len(train), 1))
        target = np.zeros((len(train), 1))
        xval_data = np.zeros((len(val), 1))
        target_val = np.zeros((len(val), 1))
valAvg = np.reshape(valAvg,(-1,1))


# In[108]:


fig = plt.figure(figsize=(8,6))
lr = [0.01, 0.1, 1, 10, 100, 1000]
plt.semilogx(lr, valAvg, color='blue')
plt.axhline(y=0, c="red", label="y=0", linestyle='--')
plt.title('Average Validation Set Error(MSE) vs Regularizer Value')
plt.xlabel('Lambda (Hyper-parameter)')
plt.ylabel('Validation Set Error')
plt.legend(['Validation Set Error','Error for Unregularized Result'], loc = "best")
plt.show()

