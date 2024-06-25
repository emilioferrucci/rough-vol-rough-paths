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
from hybrid_joint import *
from diffrax import *
import matplotlib.pyplot as plt
from controls import *

def fbm012_wrapper(args):
    rho01, rho02, rho12, grid_points, T, kappa = args
    return correlated_fbm_bm_bm(rho01, rho02, rho12, grid_points, T, 0.1, kappa)