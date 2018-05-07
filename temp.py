#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 12:20:11 2018

@author: kalyani
"""
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.utils import column_or_1d
import datetime


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    bias_term = np.ones((n_data, 1)) #define bias term
    x = np.hstack((bias_term, train_data)) #add bias term
    w = initialWeights.reshape((n_feature+1,1))   #for bias
    
    theta = sigmoid(x @ w) #calculate theta
    error = -np.sum(labeli*np.log(theta) + (1-labeli)*np.log(1-theta)) #error
    print('error:'+str(error))
    temp = np.multiply((theta - labeli) , x)
    error_grad = np.sum(temp, axis=0) #gradient of error function
    
    return error, error_grad

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    N = data.shape[0]
    bias = np.ones((N,1)) #bias term containing athetall ones
    x = np.hstack((bias,data)) #adding bias term to data
    
    label = np.zeros((N,1));   # each column is probability for class Ck

    label = sigmoid(np.dot(x, W))   # compute probabilities
    label = np.argmax(label, axis=1)    # get maximum for each class
    label = label.reshape((N,1))
    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
# =============================================================================
# for i in range(n_class):
#     labeli = Y[:, i].reshape(n_train, 1)
#     args = (train_data, labeli)
#     nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#     W[:, i] = nn_params.x.reshape((n_feature + 1,))
# 
# # Find the accuracy on Training Dataset
# predicted_label = blrPredict(W, train_data)
# print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
# 
# # Find the accuracy on Validation Dataset
# predicted_label = blrPredict(W, validation_data)
# print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
# 
# # Find the accuracy on Testing Dataset
# predicted_label = blrPredict(W, test_data)
# print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
# =============================================================================


print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
train_time = []
start_time=datetime.datetime.now()  # Starting timer to calculate the training time
# =============================================================================
# clf = SVC(kernel='linear')
# #Fit the SVM model according to the given training data.
# clf.fit(train_data, train_label.ravel())
# end_time=datetime.datetime.now() # Ending timer to calculate the training time
# time_diff=end_time-start_time
# micro_sec = time_diff.seconds*1000000+time_diff.microseconds
# train_time.append(micro_sec)
# print('\n SVM part1: linear kernel')
# #"score" returns the mean accuracy on the given test data and labels.
# print('\n Training set Accuracy:' + str(clf.score(train_data, train_label)*100) + '%')
# print('\n Validation set Accuracy:' + str(clf.score(validation_data, validation_label)*100) + '%')
# print('\n Testing set Accuracy:' + str(clf.score(test_data, test_label)*100) + '%')
# print("Train Time : "+repr(train_time))
# =============================================================================
#SVM for rbf kernel and gamma=1
# =============================================================================
# train_time = []
# start_time=datetime.datetime.now()  # Starting timer to calculate the training time
# clf = SVC(kernel='rbf', gamma=1)
# #Fit the SVM model according to the given training data.
# clf.fit(train_data, train_label.ravel())
# end_time=datetime.datetime.now() # Ending timer to calculate the training time
# time_diff=end_time-start_time
# micro_sec = time_diff.seconds*1000000+time_diff.microseconds
# train_time.append(micro_sec)
# print('\n SVM part2: rbf kernel and gamma=1')
# #"score" returns the mean accuracy on the given test data and labels.
# print('\n Training set Accuracy:' + str(clf.score(train_data, train_label)*100) + '%')
# print('\n Validation set Accuracy:' + str(clf.score(validation_data, validation_label)*100) + '%')
# print('\n Testing set Accuracy:' + str(clf.score(test_data, test_label)*100) + '%')
# print("Train Time : "+repr(train_time))
# =============================================================================

# =============================================================================
# #SVM for rbf kernel
# clf = SVC(kernel='rbf')
# #Fit the SVM model according to the given training data.
# clf.fit(train_data, train_label.ravel())
# end_time=datetime.datetime.now() # Ending timer to calculate the training time
# time_diff=end_time-start_time
# micro_sec = time_diff.seconds*1000000+time_diff.microseconds
# train_time.append(micro_sec)
# print('\n SVM part3: rbf kernel')
# #"score" returns the mean accuracy on the given test data and labels.
# print('\n Training set Accuracy:' + str(clf.score(train_data, train_label)*100) + '%')
# print('\n Validation set Accuracy:' + str(clf.score(validation_data, validation_label)*100) + '%')
# print('\n Testing set Accuracy:' + str(clf.score(test_data, test_label)*100) + '%')
# print("Train Time : "+repr(train_time))
# =============================================================================
# =============================================================================
# #SVM for different C
# for i in range(10,110,10):
#     train_time = []
#     start_time=datetime.datetime.now()  # Starting timer to calculate the training time
#     clf = SVC(kernel='rbf', C=i)
#     #Fit the SVM model according to the given training data.
#     clf.fit(train_data, train_label.ravel())
#     end_time=datetime.datetime.now() # Ending timer to calculate the training time
#     time_diff=end_time-start_time
#     micro_sec = time_diff.seconds*1000000+time_diff.microseconds
#     train_time.append(micro_sec)
#     print('\n SVM part4: rbf kernel and C:'+str(i))
#     #"score" returns the mean accuracy on the given test data and labels.
#     print('\n Training set Accuracy:' + str(clf.score(train_data, train_label)*100) + '%')
#     print('\n Validation set Accuracy:' + str(clf.score(validation_data, validation_label)*100) + '%')
#     print('\n Testing set Accuracy:' + str(clf.score(test_data, test_label)*100) + '%')
#     print("Train Time : "+repr(train_time))
# =============================================================================
#C=100
train_time = []
start_time=datetime.datetime.now()  # Starting timer to calculate the training time
clf = SVC(kernel='rbf', C=100)
#Fit the SVM model according to the given training data.
clf.fit(train_data, train_label.ravel())
end_time=datetime.datetime.now() # Ending timer to calculate the training time
time_diff=end_time-start_time
micro_sec = time_diff.seconds*1000000+time_diff.microseconds
train_time.append(micro_sec)
print('SVM part4: rbf kernel and C:'+str(100))
#"score" returns the mean accuracy on the given test data and labels.
print('Training set Accuracy:' + str(clf.score(train_data, train_label)*100) + '%')
print('Validation set Accuracy:' + str(clf.score(validation_data, validation_label)*100) + '%')
print('Testing set Accuracy:' + str(clf.score(test_data, test_label)*100) + '%')
print("Train Time : "+repr(train_time))





