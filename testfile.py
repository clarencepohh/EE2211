# -*- coding: utf-8 -*-
import numpy as np
import A1_A0254474Y as grading

X = np.array([[1, 1], [4, 2], [4, 6], [3, -6], [0, -10]])
y = np.array([[-3], [2], [1], [5], [4]])
InvXTX, w = grading.A1_A0254474Y(X, y)
print(InvXTX)
print(w)