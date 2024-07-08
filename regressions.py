
import numpy as np
from tools import extracting_initialinfo

def lasso_subgradient(func):
    #Useful when a function as the one used in lasso is not differentiable
    func = ((func < 0)* -1) + ((func > 0) * 1)
    return func


def ridge_regression(x,y,learning_rate=0.01,n_iter=1000,sigma=0.1, gradient_type="gradient_descent",normalize=None):
    #Ridge regression summing l2 weight norm to cost function of linear regression
    x, n_samples, n_features, weights = extracting_initialinfo(x,normalize)
    #sigma = parametro de regularizacion well known as alpha
    for _ in range(n_iter):
        
        if gradient_type=="gradient_descent":

            gradients = 2/(n_samples) * np.dot(x.T,( np.dot(x,weights) - y)) + (2*sigma*weights)
            weights -= learning_rate * gradients

        elif gradient_type == "coordinate_descent":
            
            for j in range(n_features):

                xi = x[:,j]

                gradient = (2/n_samples)*np.dot(xi.T,(np.dot(weights,x.T)-y))+ (2*sigma*weights[j])
            
                weights[j] -= learning_rate * gradient
        
    return weights

def lasso_regression(x,y,learning_rate=0.01,n_iter=1000,omega=0.1, gradient_type="gradient_descent",normalize=None):

    x, n_samples, n_features, weights = extracting_initialinfo(x,normalize)

    for _ in range(n_iter):
            
        if gradient_type=="gradient_descent":

            gradients = 2/(n_samples) * np.dot(x.T,( np.dot(x,weights) - y)) + (omega* lasso_subgradient(weights))
            weights -= learning_rate * gradients

        elif gradient_type == "coordinate_descent":
                
            for j in range(n_features):

                xi = x[:,j]

                gradient = (2/n_samples)*np.dot(xi.T,(np.dot(weights,x.T)-y))+ (omega*lasso_subgradient(weights[j]))
                
                weights[j] -= learning_rate * gradient
        
    return weights

def elastic_net(x,y,learning_rate=0.01,n_iter=1000,sigma=0.1,omega=0.1, gradient_type="gradient_descent",normalize=None):
    #Combination of ridge and lasso
    x, n_samples, n_features, weights = extracting_initialinfo(x,normalize)

    for _ in range(n_iter):
            
        if gradient_type=="gradient_descent":

            gradients = 2/(n_samples) * np.dot(x.T,( np.dot(x,weights) - y)) + (2*sigma*weights)+(omega* lasso_subgradient(weights))
            weights -= learning_rate * gradients

        elif gradient_type == "coordinate_descent":
                
            for j in range(n_features):

                xi = x[:,j]
                weightsi = weights[j]
                gradient = (2/n_samples)*np.dot(xi.T,(np.dot(weights,x.T)-y))+ (2*sigma*weights[j])+(omega*lasso_subgradient(weightsi))
                
                weights[j] -= learning_rate * gradient
        
    return weights

