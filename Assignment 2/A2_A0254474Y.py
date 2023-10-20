# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:23:32 2023

@author: Clarence Poh 
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import OneHotEncoder
from numpy.linalg import inv

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0254474Y(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # variables given from question
    random_state = N
    test_size = 0.7
    reg = 0.0001

    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], 
                                                        test_size=test_size, random_state=random_state)
    onehot_encoder=OneHotEncoder(sparse_output=False)
    
    reshaped_training = y_train.reshape(len(y_train), 1)
    Ytr = onehot_encoder.fit_transform(reshaped_training)
    reshaped_testing = y_test.reshape(len(y_test), 1)
    Yts = onehot_encoder.fit_transform(reshaped_testing)

    Ptrain_list = CreateRegressors(X_train, 8)
    Ptest_list = CreateRegressors(X_test, 8)

    w_list = []
    for Ptrain in Ptrain_list:
        if Ptrain.shape[0] >= Ptrain.shape[1]:
            w = inv(Ptrain.T @ Ptrain + reg*np.identity(Ptrain.shape[1])) @ Ptrain.T @ Ytr
            w_list.append(w)
        else:
            w = Ptrain.T @ inv(Ptrain @ Ptrain.T + reg*np.identity(Ptrain.shape[0])) @ Ytr
            w_list.append(w)

    y_training_argmax = []
    y_test_argmax = []
    for i in range(0,8):
        y_training_est_p = Ptrain_list[i] @ w_list[i]
        y_training_cls_p = y_training_est_p.argmax(axis=1)
        y_training_argmax.append(y_training_cls_p)

        y_test_est_p = Ptest_list[i] @ w_list[i]
        y_test_cls_p = y_test_est_p.argmax(axis=1)
        y_test_argmax.append(y_test_cls_p)
        
    error_train_array = []
    for ytrain in y_training_argmax:
        error_train_array.append(sum(ytrain != y_train))

    error_test_array = []
    for ytest in y_test_argmax:
        error_test_array.append(sum(ytest != y_test))
        
    error_train_array = np.array(error_train_array)
    error_test_array = np.array(error_test_array)

    
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array

def CreateRegressors(x, max_order):
    P = [] 

    for order in range(1, max_order+1):   
        P_current_regressors = PolynomialFeatures(order).fit_transform(x)
        P.append(P_current_regressors)

    return P