# Imports
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import *


def fun_to_array(grid, fun):
    return jax.vmap(lambda t: fun(t))(grid)

def lagged_control(fun, lag):
    return lambda s,t: fun(jnp.maximum(0,t - lag/2)) - fun(jnp.maximum(0,s - lag/2))

def join_controls(lagged, original):
    return lambda s,t: jnp.concatenate((lagged(s,t), original(s,t)))

def batch_solve(key, epsilon, dim_bm, drift, diffusion, y0, solver, t0, t1, solver_epsilon, saveat):

    vbt = VirtualBrownianTree(t0, t1+2*epsilon, tol=epsilon, shape=(dim_bm,), key=key)
    
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, vbt))

    sol = diffeqsolve(terms, solver, t0, t1, dt0 = solver_epsilon, max_steps=None, y0=y0, saveat=saveat)

    return sol

vmap_batch_solve = jax.vmap(batch_solve, in_axes=(0, None, None, None, None, None, None, None, None, None, None))

def batch_sol_to_fun(solutions, num_samples):
    V = jax.vmap(LinearInterpolation, in_axes=(None, 0))(solutions.ts[0,:], solutions.ys)
    return lambda t: jax.vmap(lambda interp, t: interp.evaluate(t))(V, jnp.full((num_samples,), t))