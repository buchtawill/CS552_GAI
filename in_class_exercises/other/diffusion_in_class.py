import numpy as np


def ELBO(x, betas, t):
    sum = 0
    
    for ...:
        
        z = 
        mu_z_t_minus_1 = P(t, z_t)
    
    
    
def sample():
    # sample from normal distribution
    # Estimate x from the thing
    M = 10
    T = 100
    z = np.random.randn(M, 1)
    for t in range(T):
        mu_z_t_minus_1 = P(t, z)
        z = mu_z_t_minus_1 + randn(M)
        
    return z