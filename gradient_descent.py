import numpy as np

def batch_gradient_descent(X, y, learning_rate=0.01, n_iter=1000, regression_type="linear_regression"):
    X = np.vstack((np.ones(X.shape[0]),X))
    n_samples, n_features = X.T.shape
    weights = np.zeros(n_features)
    for _ in range(n_iter):
        # Compute the gradients
        if regression_type == "linear_regression":
            
            gradients = 2/(n_samples) * np.dot(X,( np.dot(X.T,weights) - y))
        elif regression_type == "logistic":
            gradients = (sigmod(weights.T.dot(X)) - y) /(2*n_samples)
        # Update the weights
        weights -= learning_rate * gradients
    return weights
def learning_rate_schedule(initial_learning_rate, t, decay=0.01):
    return initial_learning_rate / (1 + decay * t)
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_iter=1000, regression_type='linear_regression'):
    "It is faster and takes only one instance of the batch"
    ones = np.ones(X.shape[0])
    x_n = np.vstack((ones,X)) ## Intercept
    n_samples, n_features = x_n.T.shape
    weights = np.zeros(n_features) ## +1 because intercept
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