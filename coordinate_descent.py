import numpy as np

def coordinate_descent(X,y,learning_rate=0.01, n_iter=1000, regression_type="linear_regression"):
    
    #Coordinate descent iterates per weight and take the other as constants

    ones = np.ones(X.shape[0])
    x_n = np.vstack((ones,X)) ## Intercept
    n_samples, n_features = x_n.T.shape
    weights = np.zeros(n_features)

    for _ in range(n_iter):
        for i in range(n_features):

            xi = x_n[i,:].T
            
            if regression_type == "linear_regression":

                gradient = (2/n_samples)*np.dot(xi,(np.dot(weights,x_n)-y))
            
                weights[i] -= learning_rate * gradient
    return weights