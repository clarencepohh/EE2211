import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import OneHotEncoder

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
    
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
                                                        iris_dataset['target'], 
                                                        test_size, 
                                                        random_state)
    onehot_encoder=OneHotEncoder(sparse=False)
    
    reshaped_training = y_train.reshape(len(y_train), 1)
    Ytr = onehot_encoder.fit_transform(reshaped_training)
    reshaped_testing = y_test.reshape(len(y_test), 1)
    Yts = onehot_encoder.fit_transform(reshaped_testing)

    Ptrain_list = [PolynomialFeatures(order).fit_transform(X_train) for order in range(1, 9)]
    Ptest_list = [PolynomialFeatures(order).fit_transform(X_test) for order in range(1, 9)]

    w_list = []

    for Ptrain in Ptrain_list:
        # from Note 2: If the number of rows in the training polynomial 
        #              is less than or equal to the number of columns, 
        #              use the dual form of ridge regression.
        #              If not, use the primal form.

        if X_train.shape[0] <= X_train.shape[1]: # dual form with weight-decay L2 regularization, lambda = 0.0001
            pass

        else: # primal form with weight-decay L2 regularization, lambda = 0.0001
            pass

    

            
    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
