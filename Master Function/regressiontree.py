# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:25:36 2023

@author: Clarence
"""
import numpy as np 

def regressiontree():
    print("\n Input the total number of data samples.")
    total_samples = int(input())
    
    dataset = np.zeros(total_samples)
    print("\n Input the features one by one.")
    for i in range(0, total_samples):
        dataset[i] = input()
    
    total_sum = 0
    for i in range(0, total_samples):
        total_sum += dataset[i]
        
    mean = total_sum / total_samples
    
    dataset_meansquared = np.zeros(total_samples)
    for i in range(0, total_samples):
        dataset_meansquared[i] = (dataset[i] - mean)**2
    
    MS = 0
    for i in range(0, total_samples):
        MS += dataset_meansquared[i]
    
    MSE = MS / total_samples
    
    print("\n MSE at this level is: ", MSE)
    print("\n Do take note that these are calculations for individual levels.")
    print("\n To calculate the MSE for each level, ")
    print("\n weighted average of the MSE at each segment is required. \n \n ")
    
    print("\n Do another round of MSE calculation? (y for yes, t for total MSE, any other character otherwise)")
    user_input = input()
    
    if user_input =='y':
        regressiontree()
    elif user_input == 't':
        totalMSE()
        return
    else:
        return
    
def totalMSE():
    print("\n Input the number of MSEs available.")
    num_MSE = int(input())

    MSE_array = np.zeros((num_MSE, 2))
    print("\n Total number of samples in the level.")
    total_samples = int(input())
    
    for i in range(0, num_MSE):
        print("\n Input the MSE and its weighted average.")
        print("\n MSE: ")
        MSE_array[i][0] = float(input())
        print("\n Number of samples used to calculate this MSE: ")
        mse_samples = int(input())
        MSE_array[i][1] = mse_samples / total_samples
        
    total_MSE = 0
    for i in range(0, num_MSE):
        total_MSE += (MSE_array[i][0] * MSE_array[i][1])
    
    print("\n Overall MSE at this level: ", total_MSE)
