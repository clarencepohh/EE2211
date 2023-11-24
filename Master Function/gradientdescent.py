# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:36:42 2023

@author: Clarence
"""

def gradientdescent():
    print("\n Ensure that the function to be minimised is updated inside the source code.")
    print("\n Input initial value of variable. \n",
          "Do edit the function if more than one variable needs to be initialized. \n")
    a = float(input())
    
    print("\n Input the number of iterations.")
    num_iters = int(input())
    
    print("\n Input the desired learning rate.")
    learning_rate = float(input())
    
    ############################## TO CHANGE IF NEEDED ########################################
    # cost_function = x^4
    # differentiated cost_function = 4x^3
    def gradient(a): 
        grad = 4*a**3
        return grad
    ###########################################################################################
    
    for i in range(0, num_iters):
        grad = gradient(a)
        a = a - learning_rate*grad 
        
    print("\n After ", num_iters, " number of iterations,")
    print("\n value of a is: ", a)
    
    print("\n Do another round of gradient descent? (y for yes, any other character otherwise)")
    user_input = input()
    
    if user_input =='y':
        gradientdescent()
    else: 
        return