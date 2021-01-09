"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    gaussian_density = 0
    sum_gaussian_density= 0
    log_likelihood = 0
    sse=0
    for i in range(n):
        for j in range(K):
            X_filter = X[i]!=0 #creates array([ True, False,  True, False,  True])
            X_temp = (X[i])[X_filter] # creates array([1, 2, 3])
            mu_temp = (mixture.mu[j])[X_filter] # creates array([1, 3, 5])
            d_temp = X_temp.shape[0] # outputs 3
            sse = np.transpose((X_temp - mu_temp)) @ ((X_temp - mu_temp))
            sum_gaussian_density +=np.exp(np.log(mixture.p[j]+1e-16)-np.log((2*np.pi*mixture.var[j]))*(d_temp/2)-0.5*(1/mixture.var[j])*sse)
            sse=0
        for j in range(K):
            X_filter = X[i]!=0 #creates array([ True, False,  True, False,  True])
            X_temp = X[i][X_filter] # creates array([1, 2, 3])
            mu_temp = mixture.mu[j][X_filter] # creates array([1, 3, 5])
            d_temp = X_temp.shape[0] # outputs 3
            sse = np.transpose((X_temp - mu_temp)) @ ((X_temp - mu_temp))
            gaussian_density =np.log((mixture.p[j]+1e-16))-np.log((2*np.pi*mixture.var[j]))*(d_temp/2)-(0.5*(1/mixture.var[j])*sse)
            sse=0
            post[i][j] = np.exp(gaussian_density-np.log(sum_gaussian_density))
        log_likelihood+=np.log(sum_gaussian_density)
        gaussian_density = 0
        sum_gaussian_density= 0
            

    return post, log_likelihood
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    gaussian_density = 0
    sum_gaussian_densities= np.zeros((K,1))
    log_likelihood = 0
    sse=0
    for i in range(n):
        for j in range(K):
            X_filter = X[i]!=0 #creates array([ True, False,  True, False,  True])
            X_temp = (X[i])[X_filter] # creates array([1, 2, 3])
            mu_temp = (mixture.mu[j])[X_filter] # creates array([1, 3, 5])
            d_temp = X_temp.shape[0] # outputs 3
            sse = np.transpose((X_temp - mu_temp)) @ ((X_temp - mu_temp))
            sum_gaussian_densities[j] = (np.log(mixture.p[j]+1e-16) - np.log((2*np.pi*mixture.var[j]+1e-16))*(d_temp/2) - 0.5*(1/mixture.var[j])*sse)
            sse=0
        for j in range(K):
            X_filter = X[i]!=0 #creates array([ True, False,  True, False,  True])
            X_temp = X[i][X_filter] # creates array([1, 2, 3])
            mu_temp = mixture.mu[j][X_filter] # creates array([1, 3, 5])
            d_temp = X_temp.shape[0] # outputs 3
            sse = np.transpose((X_temp - mu_temp)) @ ((X_temp - mu_temp))
            gaussian_density =np.log((mixture.p[j]+1e-16))-np.log((2*np.pi*mixture.var[j]+1e-16))*(d_temp/2)-(0.5*(1/mixture.var[j])*sse)
            sse=0
            post[i][j] = np.exp(gaussian_density-logsumexp(sum_gaussian_densities))
        log_likelihood+=logsumexp(sum_gaussian_densities)
        gaussian_density = 0
        sum_gaussian_densities= np.zeros((K,1))
            

    return post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    """
    _,d = X.shape
    A=X>0
    n, K = post.shape
    n_hat = post.sum(axis=0)
    p = n_hat / n
    mu_hat = mixture.mu
    var = mixture.var
    deno= np.matmul(np.transpose(post),A.sum(axis=1))
    sum_numerator = 0
    sum_denominator = 0
    sse=0
    for j in range(K):
        for i in range(d):
            for u in range(n):
                if X[u][i]!=0:
                    sum_numerator+=post[u][j]*X[u][i]
                    sum_denominator+=post[u][j]
            if sum_denominator>=1:
                mu_hat[j][i]=sum_numerator/sum_denominator
            sum_numerator=0
            sum_denominator=0
            
    for j in range(K):
        for i in range(n):
            X_filter = X[i]!=0 #creates array([ True, False,  True, False,  True])
            X_temp = (X[i])[X_filter] # creates array([1, 2, 3])
            mu_temp = (mu_hat[j])[X_filter] # creates array([1, 3, 5])
            sse += (((X_temp-mu_temp)**2).sum(axis=0))*post[i][j]
        var[j] = sse / (deno[j])
        sse=0
        var[j]=np.maximum(var[j],0.25)
    return GaussianMixture(mu_hat,var, p)
    """
    
    _,d = X.shape
    A=X>0
    n, K = post.shape
    n_hat = post.sum(axis=0)
    p = n_hat / n
    mu_hat = mixture.mu
    var = mixture.var
    deno= np.matmul(np.transpose(post),A.sum(axis=1))
    sum_numerator = 0
    sum_denominator = 0
    sse=0
    for j in range(K):
        for i in range(d):
            for u in range(n):
                if X[u][i]!=0:
                    sum_numerator+=post[u][j]*X[u][i]
                    sum_denominator+=post[u][j]
            if sum_denominator>=1:
                mu_hat[j][i]=sum_numerator/sum_denominator
            sum_numerator=0
            sum_denominator=0
            
    for j in range(K):
        for i in range(n):
            X_filter = X[i]!=0 #creates array([ True, False,  True, False,  True])
            X_temp = (X[i])[X_filter] # creates array([1, 2, 3])
            mu_temp = (mu_hat[j])[X_filter] # creates array([1, 3, 5])
            sse += (((X_temp-mu_temp)**2).sum(axis=0))*post[i][j]
        var[j] = sse / (deno[j])
        sse=0
        var[j]=np.maximum(var[j],min_variance)
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
        mixture = mstep(X, post,mixture,0.25)

    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model
    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians
    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X=np.copy(X)
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    gaussian_density = 0
    sum_gaussian_densities= np.zeros((K,1))
    log_likelihood = 0
    sse=0
    max_value_gaussian =0
    for i in range(n):
        for j in range(K):
            X_filter = X[i]!=0 #creates array([ True, False,  True, False,  True])
            X_temp = (X[i])[X_filter] # creates array([1, 2, 3])
            mu_temp = (mixture.mu[j])[X_filter] # creates array([1, 3, 5])
            d_temp = X_temp.shape[0] # outputs 3
            sse = np.transpose((X_temp - mu_temp)) @ ((X_temp - mu_temp))
            sum_gaussian_densities[j] = (np.log(mixture.p[j]+1e-16) - np.log((2*np.pi*mixture.var[j]+1e-16))*(d_temp/2) - 0.5*(1/mixture.var[j])*sse)
            sse=0
        for j in range(K):
            X_filter = X[i]!=0 #creates array([ True, False,  True, False,  True])
            X_temp = X[i][X_filter] # creates array([1, 2, 3])
            mu_temp = mixture.mu[j][X_filter] # creates array([1, 3, 5])
            d_temp = X_temp.shape[0] # outputs 3
            sse = np.transpose((X_temp - mu_temp)) @ ((X_temp - mu_temp))
            gaussian_density =np.log((mixture.p[j]+1e-16))-np.log((2*np.pi*mixture.var[j]+1e-16))*(d_temp/2)-(0.5*(1/mixture.var[j])*sse)
            sse=0
            max_value_gaussian = np.max(sum_gaussian_densities)
            post[i][j] = np.exp(gaussian_density-max_value_gaussian-logsumexp(sum_gaussian_densities-max_value_gaussian))
        log_likelihood+=logsumexp(sum_gaussian_densities)
        gaussian_density = 0
        sum_gaussian_densities= np.zeros((K,1))
    A=(X>0)-1
    C=-1*(A* (post @ mixture.mu))
    return X+C

X = np.loadtxt('test_incomplete.txt')
K=4
m = None
max_likelihood = -50000000000
seed = 4
mixture,post=init(X,K,seed)
post,cost = estep(X,mixture)
_,_,p = mstep(X,post,mixture )
mixture_latest,_,likelihood = run(X,mixture,post)
"""
mixture_new = mstep(X,post,mixture,0.25)
    mixture_latest,_,likelihood = run(X,mixture_new,post)
    if likelihood >= max_likelihood:
        max_likelihood = likelihood
        fillmatrix = fill_matrix(X,mixture_latest)
"""
print(likelihood)
