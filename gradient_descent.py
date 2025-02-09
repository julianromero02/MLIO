import numpy as np
from tools import extracting_initialinfo, learning_rate_schedule

def batch_gradient_descent(X, y, learning_rate=0.01, n_iter=1000, regression_type="linear_regression",decay_rate=False):
    #
    X, n_samples, n_features, weights = extracting_initialinfo(X)
    for t in range(n_iter):
        # Compute the gradients
        if regression_type == "linear_regression":
            
            gradients = 2/(n_samples) * np.dot(X.T,( np.dot(weights,X.T) - y))
        elif regression_type == "logistic":
            gradients = (sigmod(weights.T.dot(X)) - y) /(2*n_samples)
        # Update the weights
        if decay_rate:
            learning_rate = learning_rate_schedule(learning_rate,t)
        weights -= learning_rate * gradients
    return weights

def stochastic_gradient_descent(X, y, learning_rate=0.01, n_iter=1000, regression_type='linear_regression'):
    "It is faster and takes only one instance of the batch"
   
    x_n, n_samples, n_features, weights =extracting_initialinfo(X)
    random_index=np.random.randint(n_samples)
    for epoch in range(n_iter):
        for i in range(n_samples):
            random_index = np.random.randint(n_samples)
            xi = x_n[:,random_index]
            yi=y[random_index]
            # Compute the gradient for the i-th sample
            if regression_type == 'logistic':
                gradient=(sigmod(weights.T.dot(xi))-yi).dot(xi)

            if regression_type == 'linear_regression':
                gradient = 2/(n_samples)*(((weights.T * xi) - yi)*xi) #Cambiar para aceptar los gradientes de diferentes funciones
            
            learning_rate = learning_rate_schedule(learning_rate,epoch )
            # Update the weights
            weights -= learning_rate * gradient
    return weights


def sigmod(z):
    return 1 / (1 + np.exp(-z))