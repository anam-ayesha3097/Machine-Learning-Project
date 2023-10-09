#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the required Libraries to perform Kernel Regression
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[2]:


#Import the loadData() and normalizeData() function from utils file
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)


# In[3]:


#Seperate the dataset and target in train(100) and test datapoints
#In this experiment use on the 3rd feature of AutMPG dataset i.e. HorsePower
x_train = X_n[0:100,3]
x_test = X_n[100:,3]

y_train = t[0:100]
y_test = t[100:]

x_train = np.reshape(x_train, (-1, 1))
x_test = np.reshape(x_test, (-1, 1))
y_train = y_train.reshape(-1, 1)


# In[4]:


# # define Epanechnikov kernel function
def epanechnikov_kernel(u, h):
    kernel_val = np.zeros_like(u)
    absu = abs(u) / h <= 1
    kernel_val[absu] = 3/4 * (1 - (u[absu]/h)**2)
    return kernel_val


# In[5]:


def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse


# In[6]:


def predict(X_train, y_train, X_test, bw):
    n_test = X_test.shape[0]
    y_pred = np.zeros(n_test)
    for i in range(n_test):
            k = epanechnikov_kernel(X_train - X_test[i], bw)
            y_pred[i] = np.sum(k * y_train) / np.sum(k)
    return y_pred


# In[36]:


kfold = KFold(n_splits=10, shuffle=True)
h = [0.01, 0.1, 0.25, 1, 2, 3, 4]

mse_list = []
bandwidth_error = []
bandwidth_error.clear()
for bw in h:
    for train_idx, val_idx in kfold.split(x_train):
        #Segregating the train and validation datasets based on the k-fold cross-validation indexes
        # Split data into training and validation sets
        X_train, X_val = x_train[train_idx], x_train[val_idx]
        Y_train, Y_val = y_train[train_idx], y_train[val_idx]
        y_pred = predict(X_train, Y_train, X_val, bw)
    bandwidth_error.append(mean_squared_error(Y_val, y_pred))


# In[37]:


fig = plt.figure(figsize=(8,6))
plt.semilogx(h, bandwidth_error, color='blue')
plt.title('Average Validation Set Error(MSE) vs Bandwidth')
plt.xlabel('Bandwidth (Hyper-parameter)')
plt.ylabel('Validation Set Error')
plt.legend(['Validation Set Error'], loc = "best")
plt.show()


# In[38]:


#Seperate the dataset and target in train(100) and test datapoints
#In this experiment use on the 3rd feature of AutMPG dataset
x_train = X_n[0:100,2]
x_test = X_n[100:,2]

y_train = t[0:100]
y_test = t[100:]

x_train = np.reshape(x_train, (-1, 1))
x_test = np.reshape(x_test, (-1, 1))
y_train = y_train.reshape(-1, 1)


# In[45]:


kfold = KFold(n_splits=10, shuffle=True)
h = [0.01, 0.1, 0.25, 1, 2, 3, 4]

mse_list = []
bandwidth_error = []
bandwidth_error.clear()
for bw in h:
    for train_idx, val_idx in kfold.split(x_train):
        #Segregating the train and validation datasets based on the k-fold cross-validation indexes
        # Split data into training and validation sets
        X_train, X_val = x_train[train_idx], x_train[val_idx]
        Y_train, Y_val = y_train[train_idx], y_train[val_idx]
        y_pred = predict(X_train, Y_train, X_val, bw)
    bandwidth_error.append(mean_squared_error(Y_val, y_pred))


# In[46]:


fig = plt.figure(figsize=(8,6))
plt.semilogx(h, bandwidth_error, color='blue')
plt.title('Average Validation Set Error(MSE) vs Bandwidth')
plt.xlabel('Bandwidth (Hyper-parameter)')
plt.ylabel('Validation Set Error')
plt.legend(['Validation Set Error'], loc = "best")
plt.show()


# In[ ]:




