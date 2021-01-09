import numpy as np
import em
import common

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")
K = 12
n, d = X.shape

# TODO: Your code here
seed =1
mixture,post=init(X,K,seed)
post,cost = estep(X,mixture)
mixture_new = mstep(X,post,mixture,0.25)
mixture_latest,_,_ = run(X,mixture_new,post)
X_pred = fill_matrix(X,mixture_latest)
print(common.rmse(X_gold,X_pred))

