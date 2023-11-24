import sys

### Below here are user-made functions, abstracted for visual clarity.
### Refer to individual files for _some_ documentation
import linalgfunc
import linalginv
import regression
import gradientdescent
import perfmetrics
import regressiontree
import confusion
import kcluster

print("\n Input 1 for linear algebra matrix related functions.")
print("\n Input 2 for checking invertibility, presence of left and right inverses for a matrix X.")
print("\n Input 3 for regression: linear and polynomial.")
print("\n Input 4 for gradient descent.")
print("\n Input 5 for performance metrics.")
print("\n Input 6 for MSE calculation of regression tree.")
print("\n Input 7 for confusion matrix related functions.")
print("\n Input 8 for K-Means cluster calculations.")
print("\n Input anything else to exit.")

user_choice = int(input()) # input the choice of the user

if user_choice == 1: 
    # checking for even-determined, underdetermined or overdetermined systems
    linalgfunc.linalgfunc()

elif user_choice == 2: 
    # checking for invertibility, presence of left and right inverses for a matrix X
    linalginv.linalginv()
 
elif user_choice == 3: 
    # linear regression & polynomial regression
    regression.regression()

elif user_choice == 4:
    # gradient descent 
    gradientdescent.gradientdescent()
    
elif user_choice == 5:
    # performance metrics: gini, entropy, misclassification rate
    perfmetrics.perfmetrics()
    
elif user_choice == 6:
    # MSE for regression tree
    regressiontree.regressiontree()
    
elif user_choice == 7:
    # confusion matrix
    confusion.confusion()
    
elif user_choice == 8:
    # K-means clustering
    kcluster.kcluster()
    
else:
    sys.exit()
        



