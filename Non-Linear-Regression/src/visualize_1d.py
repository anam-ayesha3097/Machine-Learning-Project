# some simple code snippet to illustrate how to plot a function
# you should modify y_ev

import matplotlib.pyplot as plt
import numpy as np
from utils import *

# X_train, X_test, t_train, t_test should all be 1-d, and need to be defined as well 
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)

x_train = X_n[0:100,2]
x_test = X_n[100:,2]

y_train = t[0:100]
y_test = t[100:]

X_train = x_train.flatten()
X_test = x_test.flatten()

t_train = y_train.flatten()
t_test = y_test.flatten()
# plot a curve showing learned function
x_ev = np.arange(min(X_n), max(X_n) + 0.1, 0.1)
y_ev = np.array(min(t), max(t) + 0.1, 0.1) # put your regression estimate here

plt.plot(x_ev, y_ev, 'r.-')
plt.plot(X_train, t_train, 'gx', markersize=10)
plt.plot(X_test, t_test, 'bo', markersize=10, mfc='none')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Fig degree %d polynomial' % 5)
