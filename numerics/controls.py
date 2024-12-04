##
## Ofelia's (deeply) commented version
##

# Import the needed libraries
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import *

# FUNCTIONS:


# OUTPUT: the function evaluated on the assigned grid (as a vector)
def fun_to_array(grid, fun): 
    """
    grid: simulation grid
    fun: function
    """
    return jax.vmap(lambda t: fun(t))(grid)

# OUTPUT: a function of two arguments (s & t) which is the difference of the input function evaluated at the positive part of t-lag and s-lag
def lagged_control(fun, lag): 
    # I should have written this to accept a function of two parameters, but let's continue like this
    """
    lag: the lag (delay)
    fun: function
    """
    return lambda s,t: fun(jnp.maximum(0,t - lag)) - fun(jnp.maximum(0,s - lag))

# OUTPUT: a vector function which is the concatenation of the two input functions
def join_controls(lagged, original):
    """
    lagged, original: two functions of two arguments
    """
    return lambda s,t: jnp.concatenate((lagged(s,t), original(s,t)))

def stack_controls(lagged, original):
    return lambda s,t: jnp.vstack((lagged(s,t), original(s,t)))

 
# OUTPUT: 
def batch_sol_to_fun(solutions, num_samples):
    """
    solutions, 
    num_samples
    """
    V = jax.vmap(LinearInterpolation, in_axes=(None, 0))(solutions.ts[0,:], solutions.ys)
    return lambda t: jnp.squeeze(jax.vmap(lambda interp, t: interp.evaluate(t))(V, jnp.full((num_samples,), t)))

 
# OUTPUT: 
def evaluate_and_reshape(interp):
    """
    interp
    """
    return lambda s,t: jnp.atleast_1d(interp(s,t))

def evaluate_and_reshape_fun(fun):
    return lambda t: jnp.atleast_1d(fun(t))


def batch_solve(key, epsilon, dim_bm, drift, diffusion, y0, solver, t0, t1, solver_epsilon, saveat, args=None):
    vbt = VirtualBrownianTree(t0, t1+2*epsilon, tol=epsilon, shape=(dim_bm,), key=key)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, vbt))
# Function to solve the SDE 
# OUTPUT: solution to the SDE
def batch_solve(key, epsilon, dim_bm, drift, diffusion, y0, solver, t0, t1, solver_epsilon, saveat, args=None):
    """
    key = random seeed?,       
    epsilon = lag?, 
    dim_bm = dim of the Bm, 
    drift = drift term in the SDE, 
    diffusion = drift term in the SDE, 
    y0 = initial value term in the SDE, 
    solver = type of solve for the SDE we want to use, 
    t0, t1 = initial and terminal time, 
    solver_epsilon = time step used to solve the SDE, 
    saveat = ?
    """
    # Brownian (of dim dim_bm) simulation that discretises the interval [t0, t1+2*epsilon] to tolerance tol (for a fixed random seed)
    vbt = VirtualBrownianTree(t0, t1+2*epsilon, tol=epsilon, shape=(dim_bm,), key=key)    
    # putting together drift and diffusion of the SDE
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, vbt))
    # solving the SDE
    sol = diffeqsolve(terms, solver, t0, t1, dt0 = solver_epsilon, max_steps=None, y0=y0, saveat=saveat, args = args)
    return sol

# OUTPUT: ??
vmap_batch_solve = jax.vmap(batch_solve, in_axes=(0, None, None, None, None, None, None, None, None, None, None))


# this is for batch solving when we need to vary the diffusion coefficient in the price, because it is dependent on the previously solved vol
vmap_batch_solve_diff_temp = jax.vmap(batch_solve, in_axes=(0, None, None, None, None, None, None, None, None, None, None, 0))
def vmap_batch_solve_diff(split_key, epsilon, dim_bm, drift, diffusion, y0, solver, t0, t1, solver_epsilon, saveat):
    return vmap_batch_solve_diff_temp(split_key, epsilon, dim_bm, drift, diffusion, y0, solver, t0, t1, solver_epsilon, saveat, jnp.arange(split_key.shape[0]))

def batch_fbm_solve(sample, control, drift, diffusion, y0, solver, t0, t1, solver_epsilon, saveat, args=None):
    sample_control = lambda s,t: control(s,t)[:,sample]
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, sample_control))
    return diffeqsolve(terms, solver, t0, t1, dt0 = solver_epsilon, max_steps=None, y0=y0, saveat=saveat, args = args)

vmap_batch_fbm_solve = jax.vmap(batch_fbm_solve, in_axes=(0, None, None, None, None, None, None, None, None, None))

