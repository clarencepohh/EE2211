import numpy as np
from numpy.linalg import inv

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0254474Y(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
   
    """

    # your code goes here
    InvXTX = inv(X.T @ X)
    w = inv(X.T @ X) @ X.T @ y
    
    # return in this order
    return InvXTX, w
