"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    gaussian_density = 0
    sum_gaussian_density= 0
    counter = 0
    for i in range(n):
        for j in range(K):
            sse = np.transpose((X[i] - mixture.mu[j])) @ ((X[i] - mixture.mu[j]))
            sum_gaussian_density +=mixture.p[j]*(1/(2*np.pi*mixture.var[j])**(d/2))*np.exp(-0.5*(1/mixture.var[j])*sse)
        for j in range(K):
            sse = np.transpose((X[i] - mixture.mu[j])) @ ((X[i] - mixture.mu[j]))
            gaussian_density = mixture.p[j]*(1/(2*np.pi*mixture.var[j])**(d/2))*np.exp(-0.5*(1/mixture.var[j])*sse)
            post[i][j] = gaussian_density/sum_gaussian_density
        counter+=np.log(sum_gaussian_density)
        gaussian_density = 0
        sum_gaussian_density= 0
            

    return post, counter


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    _,d = X.shape
    n, K = post.shape
    n_hat = post.sum(axis=0)
    p = n_hat / n
    mu_hat = np.zeros((K,d))
    var = np.zeros(K)
    for j in range(K):
        mu_hat[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu_hat[j] - X)**2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])
    return GaussianMixture(mu_hat,var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or cost - prev_cost >= (1e-6)*np.abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, cost


X = np.loadtxt('toy_data.txt')
n,_ = X.shape
K=3
seed=1
mixture,post=init(X,K,seed)
_,_,cost = run(X,mixture,post)
print(cost)
print(bic(X,mixture,cost))
