import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    return ((X @ np.transpose(Y))+c)**p



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    '''square_x = np.sum(np.square(X))
    square_y = np.sum(np.square(Y))
    final = square_x + square_y - 2 * (np.dot(X, np.transpose(Y)))
    return np.exp(-gamma*final)
    '''
    K = np.zeros((np.size(X,0),np.size(Y,0)))
    for i in range(np.size(X,0)):
        for j in range(np.size(Y,0)):
            K[i][j] = np.exp((-1*gamma)*(np.sum((X[i]-Y[j])**2)))
    return K