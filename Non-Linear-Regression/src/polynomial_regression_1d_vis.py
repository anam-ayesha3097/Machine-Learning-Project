#!/usr/bin/env python
# coding: utf-8

# In[22]:


#Importing all the required Libraries to perform polynomial Regression without Regularization
import numpy as np
import matplotlib.pyplot as plt
from utils import *


# In[23]:


#Import the loadData() and normalizeData() function from utils file
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)


# In[24]:


#Seperate the dataset and target in train(100) and test datapoints
#In this experiment use on the 3rd feature of AutMPG dataset i.e. HorsePower
x_train = X_n[0:100,2]
x_test = X_n[100:,2]

y_train = t[0:100]
y_test = t[100:]

x_train = np.reshape(x_train, (-1, 1))
x_test = np.reshape(x_test, (-1, 1))


# In[25]:


#MSE(Mean Squared Error) Loss function
def loss(y_target, y_hyp):
    #MSE - Mean((target - hypothesis)**2)
    loss = np.square(np.subtract(y_target, y_hyp)).mean()
    return loss


# In[26]:


#Least Square Method for calculating the weights ans biases
def leastSquareMethod(phiX, target):
    lsw = np.dot(np.dot(np.linalg.inv(np.dot(phiX.T, phiX)),phiX.T),target)
    lsb = np.subtract(target, np.dot(phiX, lsw))
    return lsw, lsb


# In[27]:


def trainLSQ(data, target, degree):
    #Train model using Polynomial Regression(no regularization) of the form yhyp = mf(x) + c
    #where, m -> slope (w - learnable parameters)
    #       c -> intercept (bias - learnable parameter)
    
    #Passing data through the basis function
    phiX = degexpand(data, degree)
    
    #Calculating weights and biases using the Least Square Method
    weight, bias = leastSquareMethod(phiX, target) 
    
    #Calcultaing the hypothesis(model) based on the basis f(x) and weights
    y_hyp = np.dot(phiX, weight) + bias
    
    return y_hyp


# In[28]:


#Plot a curve showing learned function - used from visualize_1d
def plot(trainP, testP, degree):
    fig = plt.figure(figsize=(8,6))
    x_ev = np.arange(min(X_n[:, 2]), max(X_n[:, 2]) + 0.1, 0.1)
    #Draw the line of polynomial regression:
    y_ev = np.poly1d(np.polyfit(X_n[:, 2], t, degree))# put your regression estimate here
    plt.plot(x_ev, y_ev(x_ev),'r.-')
    plt.plot(x_train, trainP, 'gx',  markersize=10)
    plt.plot(x_test, testP, 'bo',  markersize=10, mfc='none')
    plt.legend(["Data", "Train Polynomial predictions","Test Polynomial Predictions"])
    plt.xlabel('x - data samples')
    plt.ylabel('t - target')
    plt.title('Fig degree %d polynomial' % degree)
    plt.show()


# In[29]:


#Train the model and get the train and test predicted values plot for degrees from 1 to 10)
trainP = []
testP = []

for degree in range(1,11):
    #Getting Prediction y_hyp for train dataset
    trainP = trainLSQ(x_train, y_train, degree)
    
    #Getting Prediction y_hyp for test dataset
    testP = trainLSQ(x_test, y_test, degree)
    
    #Plotting Train Predcition vs Test Prediction on degrees 1 to 10
    plot(trainP, testP, degree)
    trainP = []
    testP = []

