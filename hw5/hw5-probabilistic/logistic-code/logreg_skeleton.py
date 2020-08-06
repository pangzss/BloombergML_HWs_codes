import numpy as np
from scipy.optimize import minimize
from functools import partial
def logistic_func(inp):
    s = inp
    base = np.maximum(np.zeros_like(s),-s)
    return base + np.logaddexp(-base,-s-base)

def f_objective(theta, X, y, l2_param=1,val=False):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    n = y.shape[0]
    s = y*np.dot(X,theta)
    
    if np.all(X[:,-1]==1):
        J = (1/n)*np.sum(logistic_func(s))+l2_param*np.dot(theta[:-1],theta[:-1])*(val==False)
    else:
        J = (1/n)*np.sum(logistic_func(s))+l2_param*np.dot(theta,theta)*(val==False)
    return J

def fit_logistic_reg(X, y, objective_function=f_objective, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    assert np.all(X[:,-1]==1), "The last column of the design matrix should be all 1's (for bias terms)"
    J = partial(objective_function, X=X, y=y, l2_param=l2_param)
    w0 = np.zeros(X.shape[1])
    optimal_w = minimize(J, w0).x
    return optimal_w

if __name__ == "__main__":
    s = 100
    print(np.log(1+np.exp(-s)))
    print(logistic_func(s))
