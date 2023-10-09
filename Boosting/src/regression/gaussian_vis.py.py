#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the required Libraries to perform Kernel Regression
import numpy as np
import matplotlib.pyplot as plt
from utils import *


# In[2]:


#Import the loadData() and normalizeData() function from utils file
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)


# In[3]:


#Seperate the dataset and target in train(100) and test datapoints
#In this experiment use on the 4th feature of AutMPG dataset
x_train = X_n[0:100,3]
x_test = X_n[100:,3]

y_train = t[0:100]
y_test = t[100:]


# In[4]:


# define Gaussian kernel function
def gaussian_kernel(u, h):
    normalization = 1 / np.sqrt(2*np.pi * h**2)
    return normalization * np.exp(-u**2 / (2*h**2))


# In[5]:


# plot the training data and test data
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.scatter(x_test, y_test, color='green', label='Test data')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training and test data')
plt.show()


# In[6]:


#Plot a curve showing learned function - used from visualize_1d
def plot(y_pred, bw):
    # plot the learned regression function and test data
    plt.figure(figsize=(8, 6))
    plt.plot(x_train, y_train, 'b+', markersize=10, label='Train Data')
    plt.plot(x_test, y_pred, 'r.',markersize= 10,label='Learned function')
    plt.plot(x_test, y_test, 'gx', markersize=10, label = 'Test Data')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Kernel regression with h={}'.format(bw))
    plt.show()


# In[7]:


h = [0.01, 0.1, 0.25, 1, 2, 3, 4]


# In[8]:


# loop over h values and fit kernel regression model for each h
for bw in h:
    # compute kernel regression
    y_pred = np.zeros_like(x_test)
    for i in range(len(x_test)):
        k = gaussian_kernel(x_train - x_test[i], bw)
        y_pred[i] = np.sum(k * y_train) / np.sum(k)
    plot(y_pred, bw)
    k = []
    y_pred = []


# In[9]:


#Seperate the dataset and target in train(100) and test datapoints
#In this experiment use on the 3rd feature of AutMPG dataset i.e. HorsePower
x_train = X_n[0:100,2]
x_test = X_n[100:,2]

y_train = t[0:100]
y_test = t[100:]


# In[10]:


# plot the training data and test data
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.scatter(x_test, y_test, color='green', label='Test data')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training and test data')
plt.show()


# In[11]:


# loop over h values and fit kernel regression model for each h
for bw in h:
    # compute kernel regression
    y_pred = np.zeros_like(x_test)
    for i in range(len(x_test)):
        k = gaussian_kernel(x_train - x_test[i], bw)
        y_pred[i] = np.sum(k * y_train) / np.sum(k)
    plot(y_pred, bw)
    k = []
    y_pred = []


# In[ ]:




