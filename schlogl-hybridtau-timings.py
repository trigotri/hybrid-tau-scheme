import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import os
import sys
import matplotlib.pyplot as plt
from numba import jit

def define_system():

    S = ['S']
    emptySet = []

    c1 = 3e-7
    c2 = 1e-4
    c3 = 1e-3
    c4 = 3.5

    N1 = 1e5
    N2 = 2e5

    c1N12 = c1*N1/2
    c26 = c2/6
    c3N2 = c3*N2

    r1 = cu.Reaction(2*S, 3*S, c1)
    r2 = cu.Reaction(3*S, 2*S, c2)
    r3 = cu.Reaction(emptySet, S, c3)
    r4 = cu.Reaction(S, emptySet, c4)
    rs = cu.ReactionSet(r1, r2, r3, r4)

    V = rs.stochioimetric_matrix()
    return rs,V,(c1,c2,c3,c4,N1,N2,c1N12,c26,c3N2)

@jit(nopython=True)
def prop_fun(props : np.array, X : np.array, css : np.array):
    c1,c2,c3,c4,N1,N2,c1N12,c26,c3N2 = css

    S = X[0]
    props[0] = c1N12*S*(S-1)
    props[1] = c26*S*(S-1)*(S-2)
    props[2] = c3N2
    props[3] = c4*S

@jit(nopython=True)
def fbeta(x : float, I1 : float, I2 : float):
    if x <= I1:
        return 1.
    elif x >= I2:
        return 0.
    else:
        return (I2-x)/(I2-I1)

@jit(nopython=True)
def oneMinFBeta(x : float, I1 : float, I2 : float):
    return 1. - fbeta(x, I1, I2)

@jit(nopython=True)
def compute_betas(betas : np.array, X : np.array, I1 : float, I2: float):
    S = X[0]
    Sbeta = fbeta(S, I1, I2) #oneMinFBeta(S,I1A,I2A)

    betas[0] = Sbeta
    betas[1] = Sbeta
    betas[2] = Sbeta
    betas[3] = Sbeta


@jit(nopython=True)
def simulate_schlogl_htau_final_time(T : float, V : np.array, css : np.array, I1 : float, I2 : float, delta_t : float, Delta_t : float):
    props = np.zeros((4,))
    betas = np.zeros((4,))

    X = np.array([250.])
    t = 0.

    i = 0
    while t < T:
        prop_fun(props, X, css)
        compute_betas(betas, X, I1, I2)
        X,t = cs.hybrid1_tau_step_numba(X, t, V, props, betas, delta_t, Delta_t)

    return X

def simulate_schlogl_htau(T : float, V : np.array, css : np.array, I1 : float, I2 : float, delta_t : float, Delta_t : float):
    props = np.zeros((4,))
    betas = np.zeros((4,))

    X = np.array([250.])
    t = 0.

    #X_hist = np.zeros((save_at_ts.size,))
    X_hist = []
    t_hist = []

    i = 0
    while t < T:
        prop_fun(props, X, css)
        compute_betas(betas, X, I1, I2)
        X,t = cs.hybrid1_tau_step_numba(X, t, V, props, props, betas, delta_t, Delta_t)
        X_hist.append(X)
        t_hist.append(t)

    return X_hist, t_hist


@jit(nopython=True)
def simulate_schlogl_htau_fixed_times(T : float, V : np.array, css : np.array, I1 : float, I2 : float, delta_t : float, Delta_t : float, save_at_ts : np.array):
    props = np.zeros((4,))
    betas = np.zeros((4,))

    X = np.array([250.])
    t = 0.

    X_hist = np.zeros((save_at_ts.size,))

    i = 0
    while t < T:
        prop_fun(props, X, css)
        compute_betas(betas, X, I1, I2)
        X,t = cs.hybrid1_tau_step_numba(X, t, V, props, props, betas, delta_t, Delta_t)

        while i < save_at_ts.size and save_at_ts[i] <= t :
           X_hist[i] = X[0]
           i+=1

    return X_hist, save_at_ts

def simulate_Npaths(Npaths : int, T : float, V : np.array, css : np.array, I1 : float, I2 : float, delta_t : float, Delta_t : float):
    X_paths = []
    t_paths = []

    for _ in range(Npaths):
        Xpath,tpath = simulate_schlogl_htau(T, V, css, I1, I2, delta_t, Delta_t)
        X_paths.append(Xpath)
        t_paths.append(tpath)

    return X_paths, t_paths


def time_paths(Npaths : int, T : float, css : np.array, I1 : float, I2 : float, delta_t : float, Delta_t : float):
    low_times = np.zeros((Npaths,))
    high_times = np.zeros((Npaths,))

    simulate_schlogl_htau_final_time(T, V, css, I1, I2, delta_t, Delta_t)

    i,j = 0,0
    while i < Npaths or j < Npaths:
        t0 = time.time()
        X = simulate_schlogl_htau_final_time(T, V, css, I1, I2, delta_t, Delta_t)
        t1 =time.time()
        if X < 300 and i < Npaths:
            low_times[i] = t1 - t0
            i+=1
        elif j < Npaths:
            high_times[j] = t1 - t0
            j+=1

    return low_times, high_times

def generate_npath_fig(show=True):

    plt.rcParams.update({'font.size': 14})

    npaths = 10
    X_paths, t_paths = simulate_Npaths(npaths, 50, V, css)
    for i in range(npaths):
        plt.plot(t_paths[i], X_paths[i])
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.savefig('schlogl-paths.pdf', format='pdf')
    if show:
        plt.show()

def mini_xp_1():

    T = int(1e6)
    rs,V,css = define_system()
    delta_t, Delta_t = 1e-2,1e-3
    I1,I2 = 35.,45.
    save_at_ts = np.arange(0, T, 50)
    X_hist, t_hist = simulate_schlogl_htau_fixed_times(T, V, css, I1, I2, delta_t, Delta_t, save_at_ts)

    plt.plot(t_hist, X_hist)
    plt.show()

if __name__ == '__main__':
    T=50 
    rs,V,css = define_system() 
    delta_t, Delta_t = 1e-2,1e-3
    I1,I2 = 35.,45.
    Npaths = 500
    low, high = time_paths(Npaths, T, css, I1, I2, delta_t, Delta_t)
    print(low.mean())
    print(high.mean())





