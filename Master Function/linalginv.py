# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:44:26 2023

@author: Clarence
"""
import numpy as np
import sys
from numpy.linalg import inv 
from numpy.linalg import matrix_rank

def linalginv():
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

    # check if matrix is a square matrix

    if num_rows != num_cols:
        print("\n Matrix is not a square matrix.")
        print("\n There is no determinant for a non-square matrix.")
        print("\n We will check for existence of left and right inverses instead.")
        
        print("\n Checking for existence of right-inverse.")
        X_Xtranspose = matrix_X @ matrix_X.T

        try:
            inverse_XXT = inv(X_Xtranspose)
        except:
            print("\n Right-inverse does not exist.")
            print("\n Stopping search for right-inverse.")
        else: 
            print("\n Right-inverse exists.")
            print("\n Inverse of X_Xtranspose (Right Inverse) is: \n", inverse_XXT)

        print("\n Now checking for existence of left-inverse.")
        X_transpose_X = matrix_X.T @ matrix_X

        try:
            inverse_XTX = inv(X_transpose_X)
        except:
            print("\n Left-inverse does not exist.")
            print("\n End of program.")
            sys.exit()
        else:
            print("\n Left-inverse exists.")
            print("\n Inverse of X_transpose_X (Left Inverse) is: \n", inverse_XTX)
            sys.exit()


    else:
        print("\n Matrix is a square matrix.")
        detX = np.linalg.det(matrix_X)
        print("\n Determinant of matrix X is: ", detX)
        rankX = matrix_rank(matrix_X)
        print("\n Rank is: ", rankX)
        print("\n If the determinant is non-zero, the matrix is invertible.")
        print("\n We will check for existence of inverse.") 
        
        # checking for existence of inverse
        try:
            inverse_X = inv(matrix_X)
        except:
            print("\n Inverse does not exist.")
            sys.exit()
        else:
            print("\n Inverse exists.")
            print("\n Inverse of X is: \n", inverse_X)
            sys.exit()