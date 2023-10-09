#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt
from utils import *
import time


# In[23]:


# Maximum number of iterations. Continue until this limit, or when erro change is below tol.
max_iter = 500
tol = 0.01


# In[24]:


# Step size for gradient descent
eta = 0.003


# In[25]:


# Get X1,X2
data=loadmat('data.mat')
X1,X2=data['X1'],data['X2']


# In[26]:


# Data matrix with column of ones at end.
X = np.vstack((X1,X2))
X = np.hstack((X,np.ones((X.shape[0],1))))


# In[27]:


# Total Training Data Points
N = X.shape[0]


# In[28]:


# Target values, 0 for class 1 (datapoints X1), 1 for class 2 (datapoints X2)
t = np.vstack((np.zeros((X1.shape[0],1)),np.ones((X2.shape[0],1))))


# In[29]:


# Set up the slope-intercept figure
plt.figure(2)
plt.rcParams['font.size']=20
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])


# In[36]:


# Initialize w.
w = np.array([1., 0., 0.]).reshape(3,1)

# Error values over all iterations
e_all = np.array([])

e_all_iter = np.array([])

train_loss_iter = np.array([])

train_loss = np.array([])

loss_all = np.array([])



for iter in range(max_iter):   
    #SGD for all N data-points as mentioned in the HW2
    for i in range(0, N):
        # Compute output using current w on data-point X[i].
        y = sigmoid(w.T @ X[i].T).T
        
        e = y - t[i]
        
        # Gradient of the error
        grad_e = e * X[i]
        
        grad_e = np.reshape(grad_e, (-1,1))
            
        # Update w, *subtracking* a step in the error derivative since we are minimizing
        w_old = w
        w = w - eta*grad_e
        
        #Calculating Train Loss at each data point
        train_loss = -np.mean(t[i] * np.log(y) + (1 - t[i]) * np.log(1 - y))
        
        loss_all = np.append(loss_all,train_loss)
    
    e_all = np.append(e_all, loss_all)
    
    # Compute average training loss
    y_hyp_all = sigmoid(X @ w)
    train_loss_iter = -np.mean(t * np.log(y_hyp_all) + (1 - t) * np.log(1 - y_hyp_all))
    
    e_all_iter = np.append(e_all_iter,train_loss_iter)
        
    if 1:
        # Plot current separator and data
        plt.figure(1)
        plt.clf()
        plt.rcParams['font.size']=20
        plt.plot(X1[:,0],X1[:,1],'g.')
        plt.plot(X2[:,0],X2[:,1],'b.')
        drawSep(plt,w)
        plt.title('Separator in data space')
        plt.axis([-5,15,-10,10])
        plt.draw()
        plt.pause(1e-17)

    # Add next step of separator in m-b space
    plt.figure(2)
    plotMB(plt,w,w_old)
    plt.draw()
    plt.pause(1e-17)
    
    # Plot error over iterations
    plt.figure(3)
    plt.rcParams['font.size']=20
    plt.plot(loss_all,'b-')
    plt.xlabel('Iteration')
    plt.ylabel('neg. log likelihood')
    plt.title('Minimization using SGD with eta 0.003 for N Data Points in iteration %d' %iter)
    plt.show()
    # Print some information
    print('iter %d, negative log-likelihood %.4f, w=%s' % (iter,train_loss,np.array2string(w.T)))

    # Stop iterating if error does not change more than tol
    if iter > 0:
        if abs(train_loss-e_all[iter-1]) < tol:
            break
    
    train_loss = np.array([])
    loss_all = np.array([])
    train_loss_iter = np.array([])


# In[39]:


# Plot error over iterations
plt.figure(4)
plt.rcParams['font.size']=20
plt.plot(e_all,'b-')
plt.xlabel('Iteration')
plt.ylabel('neg. log likelihood')
plt.title('Minimization using Stochastic gradient descent for eta 0.003')

plt.show()


# In[40]:


# Plot error over iterations
plt.figure(5)
plt.rcParams['font.size']=20
plt.plot(e_all_iter,'b-')
plt.xlabel('Iteration')
plt.ylabel('neg. log likelihood')
plt.title('Minimization using Stochastic gradient descent for eta 0.003')

plt.show()

