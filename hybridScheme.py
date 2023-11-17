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

def hybrid_scheme(grid_points, M, T, H, kappa):
    """
    grid_points: number of points in the simulation grid
    kappa: Hybrid scheme parameter (equal to 1 or 2)
    H: Hurst parameter
    M: number of paths to simulate
    T: time horizon
    """
    grid = np.linspace(0,T,grid_points)
    alpha = H-0.5
    n = grid[1]
    
    ####################################
    #compute optimal sequence b_k=(b^*_k)^(H-1/2) for k > kappa+1
    aux = grid_points - kappa - 1
    k1 = np.arange(kappa) + 1
    k2 = np.arange(aux) + kappa + 1
    b_k = (np.power(k2,alpha+1) - np.power(k2-1,alpha+1))/(alpha+1)
    ####################################

    ####################################
    #create the correlated Gaussian processes
    VAR1 = (np.power(k1,2.0*H)-np.power(k1-1,2.*H))*np.power(n,2.*H)/(2.*H)
    VAR2 = n
    COV = (np.power(k1,alpha+1)-np.power(k1-1,alpha+1))*np.power(n,alpha+1)/(alpha+1)
    ####################################

    ####################################
    #create mean vector
    mean = np.zeros(kappa+1)
    ####################################
    
    ####################################
    #create covariance matrix
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
    b_k = np.power(n,alpha)*((np.power(k2,alpha+1) - np.power(k2-1,alpha+1))/(alpha+1))
    a_k = np.array([np.append(np.zeros(kappa),b_k)])
    ####################################
    
    ####################################
    #Create the correlated samples
    size = (grid_points-1)*M
    ####################################
    
    RAND = np.random.multivariate_normal(mean, cov, size)
    RAND1 = RAND[:,0:kappa]
    
    ####################################
    #If kappa>1 modify compute X1 first
    if kappa>1:
        RAND1[1:size,1] = RAND1[0:size-1,1]
        RAND1[np.arange(M)*(grid_points-1),1] = 0
        RAND1 = np.sum(RAND1,axis=1)
    ####################################
        
    RAND1 = RAND1.reshape((M,grid_points-1))
    
    ####################################
    #Compute X2 using discrete convolution and  FFT
    RAND2 = RAND[:,kappa:kappa+1].reshape((M,grid_points-1))
    del RAND
    u = signal.fftconvolve(a_k, RAND2[:,0:grid_points-1-kappa], mode='full')
    del RAND2
    ####################################
     
    #generate X(i/n) = X1(i/n)+X2(i/n) 
    path = np.zeros([M,grid_points])    
    path[:,1:grid_points] = u[:,0:grid_points-1] + RAND1
    
    return path, 


