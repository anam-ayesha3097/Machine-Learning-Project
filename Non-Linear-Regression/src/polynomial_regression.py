#!/usr/bin/env python
# coding: utf-8

# In[62]:


#Importing all the required Libraries to perform polynomial Regression without Regularization
import numpy as np
import matplotlib.pyplot as plt
from utils import *


# In[63]:


#Import the loadData() and normalizeData() function from utils file
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)


# In[64]:


#Seperate the dataset and target in train(100) and test datapoints
x_train = X_n[0:100,:]
x_test = X_n[100:,:]

y_train = t[0:100]
y_test = t[100:]


# In[65]:


#MSE(Mean Squared Error) Loss function
def loss(y_target, y_hyp):
    #MSE - Mean((target - hypothesis)**2)
    loss = np.square(np.subtract(y_target, y_hyp)).mean()
    return loss


# In[66]:


#LeastSquareMethod for calculating the weights ans biases
def leastSquareMethod(phiX, target):
    lsw = np.dot(np.dot(np.linalg.inv(np.dot(phiX.T, phiX)),phiX.T),target)
    lsb = np.subtract(target, np.dot(phiX, lsw))
    return lsw, lsb


# In[67]:


#Gradient Descent Method for calculating the weights and biases
def gradient(data, y_target, y_hyp):
    
    m = data.shape[0]
    dw = (1/m)*np.dot(data.T, (y_target - y_hyp))
    db = (1/m)*np.sum((y_target - y_hyp))
    return dw, db


# In[68]:


def train(data, target, degree):
    
    #Additional Train method to see the behaviour of the model with the bias in Least Square Method and Gradient descent method
    #No Regularization
    
    #Passing data through the basis function
    phiX = degexpand(data, degree)

    m, n = phiX.shape
    
    #Initializing the weights and biases to zeros
    w = np.ones((n,1))
    b = 1
    
    #Reshaping target
    target = target.reshape(m,1)
    
    y_hyp = np.dot(phiX, w) + b

    dw, db = gradient(phiX, target, y_hyp)
    
    #Calculating hypothesis again based on the updated learning parameters using Gradient Descent
    y_hyp = np.dot(phiX, dw) + db
    
    #Calculating loss and appending in the list
    lgd = loss(target, y_hyp)
    
    #Calculating loss using Least Square Method
    lsw, llsb = leastSquareMethod(phiX, target)

    y_hyp = np.dot(phiX, lsw) + llsb
    lls = loss(target, y_hyp)   
    
    return w, b, lgd, lls


# In[69]:


def trainLSQ(data, target, degree):
    #Train model using Polynomial Regression(no regularization) of the form yhyp = mf(x) + c
    #where, m -> slope (w - learnable parameters)
    #       c -> intercept (bias - learnable parameter)
    
    #Passing data through the basis function
    phiX = degexpand(data, degree)
    
    #Calculating the weights based on Least Square Method
    weight = np.dot(np.dot(np.linalg.inv(np.dot(phiX.T, phiX)),phiX.T),target)
    
    #Calcultaing the hypothesis(model) based on the basis f(x) and weights
    y_hyp = np.dot(phiX, weight)
    
    #Gettig=ng the loss based on the MSE loss formula
    mse = loss(target, y_hyp)
    
    return mse


# In[70]:


#Train the model to get Gradient Loss and Least Square Loss with Bias learning on degree (1, 10)
#LDG - Gradient Loss
#LSW - Least Sqaure Loss
#for train and test dataset

trainELGD = np.zeros(10)
trainELSW = np.zeros(10)
testELGD = np.zeros(10)
testELSW = np.zeros(10)

for degree in range(1,11):
    w, b, lgd, lsw = train(x_train, y_train, degree=degree)
    trainELGD[degree - 1] = lgd
    trainELSW[degree - 1] = lsw
    w, b, lgd, lsw = train(x_test, y_test, degree=degree)
    testELGD[degree - 1] = lgd
    testELSW[degree - 1] = lsw


# In[71]:


#Train the model only by calculating the weights without bias for degree(1,10)
#LSQ - Train the model without calculating the bias for train and test dataset
trainE = np.zeros(10)
testE = np.zeros(10)
for degree in range(1,11):
    trainE[degree - 1] = trainLSQ(x_train, y_train, degree)
    testE[degree - 1] = trainLSQ(x_test, y_test, degree)


# In[72]:


#Plot for MSE Error for train and test data using Gradient Descent Method with Bias
fig = plt.figure(figsize=(8,6))
plt.plot(range(1,11), trainELGD)
plt.plot(range(1,11), testELGD)
plt.ylabel("MSE")
plt.xlim(0, 11)
plt.xlabel("Polynomial Degree")
plt.legend(["Train Error","Test Error"], loc="best")
plt.title("MSE vs Polynomial Degree without Regularization using Gradient Descent Method (with Bias)")
plt.show()


# In[73]:


#Plot for MSE Error for train and test data using Least Sqaure Method with Bias
fig = plt.figure(figsize=(8,6))
plt.plot(range(1,11), trainELSW)
plt.plot(range(1,11), testELSW)
plt.ylabel("MSE")
plt.xlabel("Polynomial Degree")
plt.xlim(0, 11)
plt.legend(["Train Error","Test Error"], loc="best")
plt.title("MSE vs Polynomial Degree without Regularization using Least Sqaure Method (with Bias)")
plt.show()


# In[74]:


#Plot for MSE Error for train and test data using Least Sqaure Method without Bias
fig = plt.figure(figsize=(8,6))
plt.plot(range(1,11), trainE)
plt.plot(range(1,11), testE)
plt.ylabel("MSE")
plt.xlabel("Polynomial Degree")
plt.xlim(0, 11)
plt.legend(["Train Error","Test Error"], loc="best")
plt.title("MSE vs Polynomial Degree without Regularization using Least Sqaure Method (no Bias)")
plt.show() 

