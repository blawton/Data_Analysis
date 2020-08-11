# A program for extracting the principle components
# (vectors that capture the largest amount of the data's variance)
# of MNIST data, and plotting them as images


import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from numpy import linalg as LA

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Digit being analyzed
i = 2

Xtrain, Ytrain = mnist.train.next_batch(60000)
Xtest, Ytest = mnist.test.next_batch(10000)

X = np.concatenate((Xtrain, Xtest), axis=0)

Y = np.concatenate((Ytrain, Ytest), axis=0)

matches = np.where(Y[:,i] == 1)

X_f = X[matches, :][0]

print(X_f.shape)

mu = np.mean(X_f, axis=0)

mu = np.reshape(mu, (1, 784))

print(mu.shape)

B = X_f - np.matmul(np.ones((X_f.shape[0], 1)), mu)

print(B.shape)

Cov = (1/69000) * np.matmul(np.transpose(B), B)

w, v = LA.eigh(Cov)
print(w.shape)
print(v.shape)

v1= v[:,783]
v2= v[:,782]
v3= v[:,781]

PC1 = np.reshape(v1, (28,28))
PC2 = np.reshape(v2, (28,28))
PC3 = np.reshape(v3, (28,28))

plt.imshow(PC1)
plt.show()
plt.imshow(PC2)
plt.show()
plt.imshow(PC3)
plt.show()