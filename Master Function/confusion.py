# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:18:47 2023

@author: Clarence
"""

def confusion():
    print("\n This function will create a 2x2 confusion matrix.")
    print("\n If a confusion matrix of higher order is needed, please modify the function.")
    print("\n Please look at the function definition for clearer definition of terms.")
    
    #   CONFUSION ||    Predicted    |    Predicted 
    #    MATRIX   ||    Positive     |    Negative
    # ----------------------------------------------------
    # ----------------------------------------------------
    #   Actually  ||      TRUE       |      FALSE
    #   Positive  ||    POSITIVE     |     NEGATIVE
    # ----------------------------------------------------
    #   Actually  ||     FALSE       |      TRUE
    #   Negative  ||    POSITIVE     |     NEGATIVE
    
    print("\n Input true positive value (Predicted Positive and Actually Positive)")
    TP = int(input())
    
    print("\n Input false negative value (Predicted Negative and Actually Positive)")
    FN = int(input())
    
    print("\n Input false positive value (Predicted Positive and Actually Negative)")
    FP = int(input())
    
    print("\n Input true negative value (Predicted Negative and Actually Negative)")
    TN = int(input())
    
    TPR = TP / (TP + FN) 
    FNR = 1 - TPR
    PPV = TP / (TP + FP)
    FDR = 1 - PPV
    class_acc = (TP + TN) / (TP + FN + FP + TN)
    
    print("\n True Positive Rate, or Recall: ", TPR)
    print("\n False Negative Rate, or Miss Rate: ", FNR)
    print("\n Precision, or Positive Predictive Value: ", PPV)
    print("\n False Discovery Rate: ", FDR)
    print("\n Classification Accuracy: ", class_acc)
    
    return