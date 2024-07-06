import numpy as np
from gradient_descent import extracting_initialinfo

def coordinate_descent(X,y,learning_rate=0.01, n_iter=1000, regression_type="linear_regression"):
    
    #Coordinate descent iterates per weight and take the other as constants

    x_n, n_samples, n_features, weights = extracting_initialinfo(X)

    for _ in range(n_iter):
        for i in range(n_features):

            xi = x_n[:,i]
            
            if regression_type == "linear_regression":

                gradient = (2/n_samples)*np.dot(xi,(np.dot(weights,x_n.T)-y))
            
                weights[i] -= learning_rate * gradient
    return weights