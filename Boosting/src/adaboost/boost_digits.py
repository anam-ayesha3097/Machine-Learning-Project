import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt
from utils import *

# boosting for recognizing MNIST digits

# Load the data X and labels t
data=loadmat('digits.mat')
X,t=data['X'],data['t']
t=t.astype(int)

# X is 28x28x1000, t is 1000x1
# Each X[:,:,i] os a 28x28 image

# Subsample images to be 14x14 for speed
X=X[::2,::2,:]

# Set up target values
# 4s are class +1, all others are class -1
f4=(t==4)
n4=(t!=4)
t[f4]=1
t[n4]=-1

# 14,14,1000
nx,ny,ndata = X.shape

# Number to use as training images
ntrain = 500

# Training and test images
X_train = X[:,:,:ntrain]
t_train = t[:ntrain]
X_test = X[:,:,ntrain:]
t_test = t[ntrain:]

# Boosting code goes here
niter = 100

# Initialize the weights
weights = np.ones((1,ntrain))/ntrain
classifier = {'alpha':np.zeros(niter), 'd':np.zeros((niter,2)).astype(int), 'p':np.zeros(niter), 'theta':np.zeros(niter)}

for iter in range(niter):
    # Find the best weak learner
    d,p,theta,correct = findWeakLearner(X_train,t_train,weights)
    ###########################################################
    ##################### Fill in #############################
    ###########################################################
    error = np.sum(weights * (correct == False)) / np.sum(weights)
    alpha = 0.5 * np.log((1 - error)/error)
    
    hx = np.zeros(ntrain)
    for i in range(ntrain):
        if (p* X_train[d[0], d[1], i]) > (p*theta):
            hx[i] = 1
        else:
            hx[i] = -1
        
        weights[0, i] *= np.exp(-alpha * hx[i] * t_train[i])
        
    weights /= np.sum(weights)
    ###########################################################
    ##################### End fill in #########################
    ###########################################################
    
    classifier['alpha'][iter]=alpha
    classifier['d'][iter,:]=d
    classifier['p'][iter]=p
    classifier['theta'][iter]=theta

# Show plots of training error and test error

train_errs = evaluateClassifier(classifier,X_train,t_train)
test_errs = evaluateClassifier(classifier,X_test,t_test)

plt.figure(1)
plt.rcParams['font.size']=20
plt.plot(train_errs,'r-')
plt.plot(test_errs,'b-')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.legend(['Training error','Test error'])

visualizeClassifier(classifier,2,(nx,ny))
