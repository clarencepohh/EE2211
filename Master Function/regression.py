# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:05:21 2023

@author: Clarence
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import inv 

def regression():
    print("\n ##################")
    print("\n ### TAKE NOTE: ###")
    print("\n ##################")
    print("\n This function is unable to do predictions for linear regression without a bias / offset.")
    print("\n Predictions should always be done with a bias for accurate results.")
    print("\n If question asks for predicting without bias, please use a separate function.")
    print("\n ################## \n")
    print("\n Are the input values for matrix X integers or floats (i for integers, f for floats)?")
    input_type_X = input()

    print("\n Input the size of the matrix X starting with number of rows")
    num_rows = int(input())
    print("\n Input the number of columns.")
    num_cols = int(input())

    if input_type_X == 'i':
        # populate the matrix X with integers
        print("\n Input the matrix row by row.")
        matrix_X = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                print("\n Input the element at position ", i+1, j+1)
                matrix_X[i][j] = int(input())

    elif input_type_X == 'f':
        # populate the matrix X with floats
        print("\n Input the matrix row by row.")
        matrix_X = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                print("\n Input the element at position ", i+1, j+1)
                matrix_X[i][j] = float(input())

    print("\n Are the input values for vector Y integers or floats (i for integers, f for floats)?")
    input_type_Y = input()
    print("\n Is one-hot encoding used? (y for yes, any other character otherwise)")
    input_one_hot = input()
    
    if input_one_hot == 'y':
        onehot = True
    else:
        onehot = False
        
    if onehot:
        print("\n If one-hot encoding is to be done,")
        print("\n rows = # of variables,")
        print("\n cols = # of classes.")
        print("\n Input the number of variables (# of rows in class matrix)")
        vect_rows = int(input())
        print("\n Input the number of classes (# of cols in class matrix)")
        vect_cols = int(input())
    
    else: 
        print("\n Input the size of the vector Y starting with number of rows.")
        vect_rows = int(input())
        print("\n Input the number of columns.")
        vect_cols = int(input())

    if input_type_Y == 'i':
        # populate the vector Y with integers
        # possible to have one hot here
        vector_Y = np.zeros((vect_rows, vect_cols))
        if onehot:
            for i in range(vect_rows):
                print("\n Input class of variable ", i)
                item_class = int(input()) - 1
                for j in range(vect_cols):
                    if j == item_class:
                        vector_Y[i][j] = 1
                    else:
                        vector_Y[i][j] = 0
            
        else: 
            print("\n Input the vector Y row by row.")
            for i in range(vect_rows):
                for j in range(vect_cols):
                    print("\n Input the element at position ", i + 1, j + 1)
                    vector_Y[i][j] = int(input())

    elif input_type_Y == 'f':
        # populate the vector Y with floats
        print("\n Input the vector Y row by row.")
        vector_Y = np.zeros((vect_rows, vect_cols))
        for i in range(vect_rows):
            for j in range(vect_cols):
                print("\n Input the element at position ", i + 1, j + 1)
                vector_Y[i][j] = float(input())

    print("\n Input the order of the polynomial to be used for regression.")
    print("\n Input 1 for linear regression.")
    order = int(input()) 
    poly = PolynomialFeatures(order)
    polynomial_X = poly.fit_transform(matrix_X)

    print ("\n Is regularization being used? (y for yes, any other character otherwise)")
    regularization = input()
    if (regularization == 'y'):

        print("\n Input the value of lambda.")
        test_lambda = float(input())
    else:
        test_lambda = 0 # if lambda = zero then in the regression formula it will not be applied
        
    if polynomial_X.shape[0] > polynomial_X.shape[1]: # primal form 
        w = inv(polynomial_X.T @ polynomial_X + test_lambda*np.eye(polynomial_X.shape[1])) @ polynomial_X.T @ vector_Y
        print("\n Primal Form was used as # of rows of polynomial X > # of columns.")
        print("\n System is overdetermined.")
        
    else: # dual form
        w = polynomial_X.T @ inv(polynomial_X @ polynomial_X.T + test_lambda*np.eye(polynomial_X.shape[0])) @ vector_Y
        if polynomial_X.shape[0] == polynomial_X.shape[1]:
            print("\n Dual Form was used as # of rows of polynomial X == # of columns.")
            print("\n System is even determined.")
        else: 
            print("\n Dual Form was used as # of rows of polynomial X < # of columns.")
            print("\n System is underdetermined.")
            
    print("\n w is:\n", w)
    print("\n Calculate w T w? (y for yes, any other character otherwise).")
    input_wtw = input()
    if input_wtw =='y':
        wtw = w.T @ w 
        print("\n WTW is (Regularization term): ", wtw)
    
    print("\n Are you calculating MSE for training data? (y for yes, any other character otherwise).")
    test_MSE = input()
    if test_MSE == 'y': 
        test_matrix_X = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                test_matrix_X[i][j] = matrix_X[i][j]
        
    else: 
        print("\n Input the test values of X starting with number of rows")
        num_rows_testX = int(input())
        print("\n Input the number of columns.")
        num_cols_testX = int(input())
    
        if input_type_X == 'i':
            # populate the test matrix with integers
            print("\n Input the matrix row by row.")
            test_matrix_X = np.zeros((num_rows_testX, num_cols_testX))
            for i in range(num_rows_testX):
                for j in range(num_cols_testX):
                    print("\n Input the element at position ", i + 1, j + 1)
                    test_matrix_X[i][j] = int(input())
    
        elif input_type_X == 'f':
            # populate the test matrix with floats
            print("\n Input the matrix row by row.")
            test_matrix_X = np.zeros((num_rows_testX, num_cols_testX))
            for i in range(num_rows_testX):
                for j in range(num_cols_testX):
                    print("\n Input the element at position ", i + 1, j + 1)
                    test_matrix_X[i][j] = float(input())

    polytest = PolynomialFeatures(order)
    polynomial_testX = polytest.fit_transform(test_matrix_X)
    predicted_Y = polynomial_testX @ w
    print("\n Predicted Y is:\n", predicted_Y)
        
    if onehot:
        class_predicted_Y = np.zeros((predicted_Y.shape[0], predicted_Y.shape[1]))
        for rows in range(predicted_Y.shape[0]):
            max_value = predicted_Y[rows][0]
            max_col = 0
            for cols in range(1, predicted_Y.shape[1]):
                if max_value < predicted_Y[rows][cols]:
                    max_value = predicted_Y[rows][cols]
                    max_col = cols
            class_predicted_Y[rows][max_col] = 1
            
        print("\n Predicted Y classes are:\n", class_predicted_Y)
        
    if test_MSE == 'y':
        MSE_array = np.zeros((1, num_cols))
        for j in range(num_cols):
            for i in range(num_rows):    
                MSE_array[0][j] += ((predicted_Y[i][j] - vector_Y[i][j]) ** 2)
                
        for j in range(num_cols):
            MSE_array[0][j] /= num_rows
        print("\n MSE:  ", MSE_array)
        
    
    return