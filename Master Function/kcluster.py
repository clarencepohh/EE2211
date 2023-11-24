# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:00:08 2023

@author: Clarence
"""

import numpy as np
from sklearn.cluster import KMeans

def kcluster():
    print("\n Input the number of data points.")
    num_pts = int(input())
    
    print("\n Input the number of starting centroids.")
    num_centres = int(input())
    
    starting_pts = np.zeros((num_centres, 2))
    print("\n Populating the starting points array...")
    for i in range(num_centres):
        print("\n For centre ", i+1, "\n")
        print("\n x-coordinate is:")
        x = float(input())
        print("\n y-coordinate is:")
        y = float(input())
        starting_pts[i] = [x,y]
        
    data_pts = np.zeros((num_pts, 2))
    print("\n Populating the data points array...")
    for i in range(num_pts):
        print("\n Data point number ", i, "\n")
        print("\n x-coordinate is:")
        x = float(input())
        print("\n y-coordinate is:")
        y = float(input())
        data_pts[i] = [x,y] 
        
    print("\n Input desired number of iterations.")
    num_iters = int(input())
    
    my_kmeans = KMeans(n_clusters=num_centres, init=starting_pts, n_init=1, max_iter=num_iters).fit(data_pts)
    print("\n Final locations of centroids: \n", my_kmeans.cluster_centers_)