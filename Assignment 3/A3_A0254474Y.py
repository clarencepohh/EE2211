import numpy as np
import pandas as pd


# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0254474Y(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    a = np.array([1.5])
    b = np.array([0.3])
    c = np.array([1])
    d = np.array([2])

    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)

    for i in range(0, num_iters):
        a = a - learning_rate*(5*a**4)
        a_out[i] = a
        f1_out[i] = a**5

        b = b - learning_rate*(2*np.sin(b)*np.cos(b))
        b_out[i] = b
        f2_out[i] = np.sin(b)**2

        c = c - learning_rate*(3*c**2)
        c_out[i] = c
        d = d - learning_rate*(3*d**2+2*d*np.sin(d)*np.cos(d))
        d_out[i] = d
        f3_out[i] = c**3 + d**2*np.sin(d)

    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 

# NOTE: GRADIENT CALCULATIONS ARE BELOW
# f1(a) = a^5
# f1'(a) = 5a^4
#
# f2(b) = sin^2(b)
# f2'(b) = 2sin(b)cos(b)
#
# f3(c,d) = c^3 + d^2sin(d)
# f3'(c) = 3c^2
# f3'(c,d) = 3c^2 + 2dsin(d) + d^2cos(d)

A3_A0254474Y(0.1, 10)