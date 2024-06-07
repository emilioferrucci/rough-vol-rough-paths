import numpy as np
import scipy
import scipy.signal as signal
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.special as special
import time
import scipy.stats
from scipy.optimize import bisect
from scipy.stats import norm
import jax.numpy as jnp
import jax

def hybrid_scheme(grid_points, M, T, H, kappa):
    """
    grid_points: number of points in the simulation grid
    kappa: Hybrid scheme parameter (equal to 1 or 2)
    H: Hurst parameter
    M: number of paths to simulate
    T: time horizon
    """
    
    # the actual time-grid
    grid = np.linspace(0,T,grid_points)
    # aux-par used in place of H
    alpha = H-0.5
    # time step (T/n =\Delta)
    n = grid[1]
    
    ####################################
    #compute optimal sequence b_k=(b^*_k)^(H-1/2) for k > kappa+1
    aux = grid_points - kappa - 1
    # vector of the first (\kappa+1)nintegers
    k1 = np.arange(kappa) + 1
    # vector of the last (n- kappa - 1) integers starting from n
    k2 = np.arange(aux) + kappa + 1
    b_k = (np.power(k2,alpha+1) - np.power(k2-1,alpha+1))/(alpha+1)
    ####################################

    ####################################
    #create the correlated Gaussian processes
    # vector of the variances of the points close to the singularity
    # this has lenght kappa
    VAR1 = (np.power(k1,2.0*H)-np.power(k1-1,2.*H))*np.power(n,2.*H)/(2.*H)
    # vector of the variances of the Brownian increments
    # this is a number
    VAR2 = n
    # the vector of the non-null covariances between the two (i.e. when they are over the same interval of time)
    # NB: it has lenght kappa
    COV = (np.power(k1,alpha+1)-np.power(k1-1,alpha+1))*np.power(n,alpha+1)/(alpha+1)
    ####################################

    ####################################
    #create mean vector
    mean = np.zeros(kappa+1)
    ####################################
    
    ####################################
    #create covariance matrix of the vector (W_{i,1},W_{i,2},W_i) for every i 
    # as its explicit form does not depend on i
    # (this is enough if kappa is 1 but if not you have to compute the covariance 
    # between W_{i,1},W_{i,2}
    cov = np.diag(np.append(VAR1,VAR2)) 
    cov[0:kappa,kappa] = COV
    cov[kappa,0:kappa] = COV
    ####################################
    
    if kappa>1:
        COV1 = np.power(n,2.*alpha+1)/(alpha+1)*special.hyp2f1(1,2.*(alpha+1),alpha+2,-1.)*np.power(2.,alpha+1)
        cov[0,1] = COV1
        cov[1,0] = COV1
        
    ####################################
    #Create optimal b*_k
    # vector of the coefficients multiplying the Brownian increments
    b_k = np.power(n,alpha)*((np.power(k2,alpha+1) - np.power(k2-1,alpha+1))/(alpha+1))
    # adding kappa zeros in front of the vector of the b_k
    a_k = np.array([np.append(np.zeros(kappa),b_k)])
    
    ####################################
    
    ####################################
    #Create the correlated samples
    size = (grid_points-1)*M
    ####################################
    # this generated (W_{i,1}, W_i) for each i and for each sample paths
    RAND = np.random.multivariate_normal(mean, cov, size)
    # this extracts W_{i,1} for each i and for each sample paths
    RAND1 = RAND[:,0:kappa]
    
    
    # we do not consider this line as we are only interested in kappa=1
    ####################################
    #If kappa>1 modify compute X1 first
    if kappa>1:
        RAND1[1:size,1] = RAND1[0:size-1,1]
        RAND1[np.arange(M)*(grid_points-1),1] = 0
        RAND1 = np.sum(RAND1,axis=1)
    ####################################
    
    # make a matrix out of the previous expression for W_{i,1} all together in a vector 
    # so that each column corresponds to a trajectory (and i ranges over the rows)
    RAND1 = RAND1.reshape((M,grid_points-1))
    
    ####################################
    #Compute X2 using discrete convolution and  FFT
    
    # make a matrix out of the previous expression for W_{i} all together in a vector 
    # so that each column corresponds to a trajectory (and i ranges over the rows)
    RAND2 = RAND[:,kappa:kappa+1].reshape((M,grid_points-1))
    
    # delete RAND as we have already saved all its infos elesewhere
    del RAND
    
    u = signal.fftconvolve(a_k, RAND2[:,0:grid_points-1-kappa], mode='full')\
    
    c_k = np.array([np.ones(grid_points-1)])
    
    v = signal.fftconvolve(c_k, RAND2[:,0:grid_points-1], mode='full')\
    
    del RAND2
    ####################################
     
    #generate X(i/n) = X1(i/n)+X2(i/n) 
    pathf = np.zeros([M,grid_points])  
    pathB = np.zeros([M,grid_points])
    
    pathf[:,1:grid_points] = u[:,0:grid_points-1] + RAND1
    pathB[:,1:grid_points] = v[:,0:grid_points-1] 
    
    return pathf, pathB




def cholesky_3by3_matrix(a, b, c): # only use if no two brownians are fully correlated
    matrix = jnp.array([[1,a,b],[a,1,c],[b,c,1]])
    return jax.scipy.linalg.cholesky(matrix, lower=True)

def correlated_fbm_bm(rho, grid_points, T, H, kappa):
    pathf, pathB = hybrid_scheme(grid_points, 2, T, H, kappa)
    correlated_fbm =  rho*pathf[0,:] + jnp.sqrt(1-rho**2)*pathf[1,:]
    return correlated_fbm, pathB[0,:]

def correlated_fbm_bm_bm(rho01, rho02, rho12, grid_points, T, H, kappa):
    pathf, pathB = hybrid_scheme(grid_points, 3, T, H, kappa)
    sigma = cholesky_3by3_matrix(rho01, rho02, rho12)
    pathf_corr = jnp.dot(sigma,pathf)
    pathB_corr = jnp.dot(sigma,pathB)
    return pathf_corr[0,:], pathB_corr[1,:], pathB_corr[2,:]


# The following function returns a fully correlated fbm and bm, and a bm with arbitrary
# correlation with the first two components (cannot be done with Cholesky bc degenerate)
def degenerate_fbm_1_bm_bm(rho, grid_points, T, H, kappa):
    pathf, pathB = hybrid_scheme(grid_points, 2, T, H, kappa)
    correlated_bm = rho*pathB[0,:] + jnp.sqrt(1-rho**2)*pathB[1,:]
    return pathf[0,:], pathB[0,:], correlated_bm
    
    
# The following function returns a fbm and two identical bms correlated with the fbm
def degenerate_fbm_bm_1_bm(rho, grid_points, T, H, kappa):
    fbm, bm = correlated_fbm_bm(rho, grid_points, T, H, kappa)
    return fbm, bm, bm

def degenerate_fbm_1_bm_1_bm(grid_points, T, H, kappa):
    pathf, pathB = hybrid_scheme(grid_points, 1, T, H, kappa)
    return pathf[0,:], pathB[0,:], pathB[0,:]
