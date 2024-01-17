import numpy as np
import chem_utils as cu
import chem_step as cs
import part_filt_utils as pfu
import part_filt_algs as pfa
import time
import sys
import os
from numba import jit, config
import time
import matplotlib.pyplot as plt

'''
Produces timings for the simulation using either SSA and Hybrid Tau.
'''

@jit(nopython=True)
def setup_xp_data(sc : float):
    c_values = (2,sc,1/50,1,1/(50*sc))
    V = np.array([[1., 0., -1., 0,  -1.],
                [0., 1.,  0., -1,  19.]])
    T = 50.
    return c_values,V,T


@jit(nopython=True)
def setup_xp_data_ht(sc : float):

    c_values,V,T = setup_xp_data(sc)
    delta_t, Delta_t = 0.1, 0.01
    I1,I2 = 20,60

    return c_values,V,T,delta_t,Delta_t,I1,I2

@jit(nopython=True)
def setup_time_xp():
    N = 10
    scs = [float(10**k) for k in range(5)]
    dat = np.empty(shape=(len(scs), N), dtype='float')
    return N,scs,dat

@jit(nopython=True)
def run_simulation_ssa(sc : float):

    c_values,V,T = setup_xp_data(sc)
    c1,c2,c3,c4,c5 = c_values
    
    X_ = np.zeros((2,))
    t = 0.
    while t < T:
        a = np.array([c1, c2, c3*X_[0], c4*X_[1], c5*X_[0]*X_[1]])
        X_,t = cs.ssa_prop_step(X_, t, V, a)

@jit(nopython=True)
def fbeta(x : float, I1 : float, I2 : float):
    if x <= I1:
        return 1.
    elif x >= I2:
        return 0.
    else:
        return (I2-x)/(I2-I1)

@jit(nopython=True)
def run_simulation_hybridtau(sc : float):


    c_values,V,T,delta_t,Delta_t,I1,I2 = setup_xp_data_ht(sc)
    c1,c2,c3,c4,c5 = c_values

    X_ = np.zeros((2,))
    t_ = 0

    beta = np.zeros((5,))
    beta[0] = 1. - (1. - fbeta(X_[0],I1,I2)) # first production reaction, only depends on X1
    beta[1] = 1. - (1. - fbeta(X_[1],I1,I2)) # second production reaction, only depends on X2
    beta[2] = 1. - (1. - fbeta(X_[0],I1,I2)) # first decay reaction, only depends on X1
    beta[3] = 1. - (1. - fbeta(X_[1],I1,I2)) # second decay reaction, only depends on X2
    beta[4] = 1. - (1. - fbeta(X_[1],I1,I2))*(1. - fbeta(X_[0],I1,I2)) # dimerization

    while t_ < T:
        X_rint = np.rint(X_)
        a = np.array([c1, c2, c3*X_[0], c4*X_[1], c5*X_[0]*X_[1]])
        a_rint = np.array([c1, c2, c3*X_rint[0], c4*X_rint[1], c5*X_rint[0]*X_rint[1]])
        X_,t_ = cs.hybrid1_tau_step_numba(X_, t_, V, a, a_rint, beta, delta_t, Delta_t)


def time_N_simulations(N : int, sc : float, simulator):
    times = np.empty((N,), dtype='float')
    for i in range(N):
        t0 = time.time()
        simulator(sc)
        t1 = time.time()
    times[i] = t1 - t0
    return times


def time_simulations(simulator):
    N,scs,dat = setup_time_xp()
    simulator(1.)
    for i,sc in enumerate(scs):
        dat[i] = time_N_simulations(N, sc, simulator)
    return scs,dat[:,1:]


if __name__ == '__main__':

    scs,dat = time_simulations(run_simulation_hybridtau)
    plt.plot(scs, dat.mean(axis=1)); plt.grid(); plt.show()
