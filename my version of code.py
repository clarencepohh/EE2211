import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import inv 
from numpy.linalg import matrix_rank

print("\n Input 1 for linear algebra matrix related functions.")
print("\n Input 2 for checking invertibility, presence of left and right inverses for a matrix X.")
print("\n Input 3 for linear regression.")
print("\n Input anything else to exit.")
user_choice = int(input()) # input the choice of the user

if user_choice == 1: # checking for even-determined, underdetermined or overdetermined systems
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
    print("\n Input the size of the vector Y starting with number of rows.")
    vect_rows = int(input())
    print("\n Input the number of columns.")
    vect_cols = int(input())
        
    if input_type_Y == 'i':
        # populate the vector Y with integers
        print("\n Input the vector Y row by row.")
        vector_Y = np.zeros((vect_rows, vect_cols))
        for i in range(vect_rows):
            for j in range(vect_cols):
                print("\n Input the element at position ", i+1, j+1)
                vector_Y[i][j] = int(input())  
                
    elif input_type_Y == 'f':
        # populate the vector Y with floats
        print("\n Input the vector Y row by row.")
        vector_Y = np.zeros((vect_rows, vect_cols))
        for i in range(vect_rows):
            for j in range(vect_cols):
                print("\n Input the element at position ", i+1, j+1)
                vector_Y[i][j] = float(input())  
                
    # now check if the system is even-determined, underdetermined or overdetermined
    system_determined = None
    if num_rows == num_cols: # even-determined
        system_determined = 1
        print("\n The system is even-determined.")
        rankX = matrix_rank(matrix_X)
        rankXY = matrix_rank(np.concatenate((matrix_X, vector_Y), axis=1))
        print("\n Rank of matrix X is: ", rankX)
        print("\n Rank of matrix [X Y] is: ", rankXY)
        detX = np.linalg.det(matrix_X)
        print("\n Determinant of matrix X is: ", detX)
        print("\n If the determinant is non-zero, the system is invertible.")
        print("\n If the determinant is zero, the system is singular.")

        if rankX == rankXY:
            print("\n System has a unique solution as rankX = rankXY.")
            print("\n We will try to find the solution using the inverse if it exists.")

            # checking for existence of inverse
            try:
                inverse_X = inv(matrix_X)
            except:
                print("\n Inverse does not exist and determinant was probably zero above.")
                print("\n End of program.")
                sys.exit()
            else:
                print("\n Inverse exists, and determinant was probably non-zero above.")
                print("\n Inverse of X is: \n", inverse_X)

                # check if there is a need to solve for w
                print("\n Solve for w given Xw=Y? (y/n)")
                user_choice = input()
                if user_choice == 'y':
                    w = inverse_X.dot(vector_Y)
                    print("\n Solution is: \n", w)
                else:
                    print("\n End of program.")
                    sys.exit()

    elif num_rows < num_cols: # underdetermined
        system_determined = 2
        print("\n The system is underdetermined.")
        print("\n There is no determinant for a non-square matrix.")
        rankX = matrix_rank(matrix_X)
        rankXY = matrix_rank(np.concatenate((matrix_X, vector_Y), axis=1))
        print("\n Rank of matrix X is: ", rankX)
        print("\n Rank of matrix [X Y] is: ", rankXY)

        if rankX == rankXY:
            print("\n System has infinitely many solutions as rankX = rankXY.")
            print("\n We will try to find a constrained solution using the right-inverse if it exists.")

            # checking for existence of right-inverse
            X_Xtranspose = matrix_X @ matrix_X.T

            try:
                inverse_XXT = inv(X_Xtranspose)
            except:
                print("\n Right-inverse does not exist.")
                print("\n End of program.")
                sys.exit()
            else:
                print("\n Right-inverse exists.")
                print("\n Inverse of X_Xtranspose (Right Inverse) is: \n", inverse_XXT)

                # check if there is a need to solve for w
                print("\n Solve for w given Xw=Y? (y/n)")
                user_choice = input()
                if user_choice == 'y':
                    X_dagger = matrix_X.T @ inverse_XXT
                    w = X_dagger.dot(vector_Y)
                    print("\n Solution is: \n", w)
                else:
                    print("\n End of program.")
                    sys.exit()
        
        elif rankX < rankXY:
            print("\n System has no solution as rankX < rankXY.")
            print("\n X_Xtranspose and X_transpose_X are not invertible.")
            print("\n End of program.")
            sys.exit()
        
    else: # overdetermined
        system_determined = 3
        print("\n The system is overdetermined.")
        print("\n There is no determinant for a non-square matrix.")
        rankX = matrix_rank(matrix_X)
        rankXY = matrix_rank(np.concatenate((matrix_X, vector_Y), axis=1))
        print("\n Rank of matrix X is: ", rankX)
        print("\n Rank of matrix [X Y] is: ", rankXY)

        if rankX == rankXY:
            print("\n System has infinitely many solutions as rankX = rankXY.")
            print("\n End of program.")
            sys.exit()

        elif rankX < rankXY:
            print("\n System has no solution as rankX < rankXY.")
            print("\n We will try to find a constrained solution using the left-inverse if it exists.")

            # checking for existence of left-inverse
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
                
                # check if there is a need to solve for w
                print("\n Solve for w given Xw=Y? (y/n)")
                user_choice = input()
                if user_choice == 'y':
                    X_dagger = inverse_XTX @ matrix_X.T
                    w = X_dagger.dot(vector_Y)
                    print("\n Solution is: \n", w)
                else:
                    print("\n End of program.")
                    sys.exit()

elif user_choice == 2: # checking for invertibility, presence of left and right inverses for a matrix X
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
 
elif user_choice == 3: # linear regression

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
    print("\n Input the size of the vector Y starting with number of rows.")
    vect_rows = int(input())
    print("\n Input the number of columns.")
    vect_cols = int(input())

    if input_type_Y == 'i':
        # populate the vector Y with integers
        print("\n Input the vector Y row by row.")
        vector_Y = np.zeros((vect_rows, vect_cols))
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
    print("\n Input 1 if linear regression is to be used.")
    order = int(input()) 
    poly = PolynomialFeatures(order)
    polynomial_X = poly.fit_transform(matrix_X)

    if polynomial_X.shape[0] > polynomial_X.shape[1]:
        w = inv(polynomial_X.T @ polynomial_X) @ polynomial_X.T @ vector_Y
        print("\n w is:\n", w)

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

else:
    sys.exit()
        



