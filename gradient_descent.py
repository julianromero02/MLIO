import numpy as np

def batch_gradient_descent(X, y, learning_rate=0.01, n_iter=1000, regression_type="linear_regression"):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    for _ in range(n_iter):
        # Compute the gradients
        if regression_type == "linear_regression":
            
            gradients = 2/np.dot(X.T, X.dot(weights) - y) / n_samples
        elif regression_type == "logistic":
            gradients = (sigmod(weights.T.dot(X)) - y) / n_samples
        # Update the weights
        weights -= learning_rate * gradients
    return weights
def learning_rate_schedule(initial_learning_rate, t, decay=0.1):
    return initial_learning_rate / (1 + decay * t)
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_iter=1000, regression_type='linear_regression'):
    "It is faster and takes only one instance of the batch"
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    random_index=np.random.randint(n_samples)
    for epoch in range(n_iter):
        for i in range(n_samples):
            random_index = np.random.randint(n_samples)
            xi = X[random_index:random_index + 1]
            yi=y[random_index:random_index + 1]
            # Compute the gradient for the i-th sample
            if regression_type == 'logistic':
                gradient=(sigmod(weights.T.dot(xi))-yi).dot(xi)
            if regression_type == 'linear_regression':
                gradient = 2/ np.dot(xi.T, xi.dot(weights) - yi) / n_samples #Cambiar para aceptar los gradientes de diferentes funciones
            learning_rate = learning_rate_schedule(learning_rate,epoch )
            # Update the weights
            weights -= learning_rate * gradient
    return weights


def sigmod(z):
    return 1 / (1 + np.exp(-z))