# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:02:35 2023

@author: Clarence
"""

import numpy as np

def perfmetrics():
    
    print("\n How many samples are there total?")
    total_samples = int(input())
    
    print("\n How many different classes are there?")
    num_classes = int(input())

    probability_matrix = np.zeros(num_classes)
    
    for i in range(0, num_classes):
        print("\n How many of class", i, "are there?")
        class_i = int(input())
        probability_matrix[i] = class_i / total_samples
    
    # metrics calculation
    gini_coeff = 1
    entropy = 0
    misclass = 1
    max_class = probability_matrix[0]
    
    for i in range(0, num_classes):
        gini_coeff -= probability_matrix[i]**2
        entropy -= probability_matrix[i] * np.log2(probability_matrix[i])
        if max_class < probability_matrix[i]:
            max_class = probability_matrix[i]
    misclass = 1 - max_class

    print("\n Performance metrics at this node:")
    print("\n Gini coefficient: ", gini_coeff)
    print("\n Entropy: ", entropy)
    print("\n Misclassification Rate: ", misclass)            
            
    print("\n Do take note that these are calculations for individual nodes.")
    print("\n To calculate the relevant metrics for node level, ")
    print("\n weighted average of the metric at each node is required. \n \n ")
    
    print("\n Do another round of metrics calculation? (y for yes, any other character otherwise)")
    user_input = input()
    
    if user_input =='y':
        perfmetrics()
    else: 
        return