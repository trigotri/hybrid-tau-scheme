import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import sys
from numba import jit
import matplotlib.pyplot as plt
# Produces files of format X{k}-{proc}.npy and corresponding time index
# using SSA for Lotka-Volterra system from paper.

## set reactions

def define_system():
    A, B = ['A'], ['B']
    k1,k2,k3 = 2.,2e-3,2.

    r1 = cu.Reaction(A, 2*A, k1)
    r2 = cu.Reaction(A+B, 2*B, k2)
    r3 = cu.Reaction(B, [], k3)
    rs = cu.ReactionSet(r1, r2, r3)

    V = rs.stochioimetric_matrix()

    return rs,V,(k1,k2,k3)

@jit(nopython=True)
def prop_fun(props : np.array, X : np.array, ks : np.array):
    k1,k2,k3 = ks
    A,B = X[0],X[1]
    props[0] = k1 * A
    props[1] = k2 * A * B
    props[2] = k3 * B

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
def compute_betas(betas : np.array, X : np.array, I1A : float, I2A: float, I1B: float, I2B : float):

    A,B= X[0],X[1]

    Abeta = oneMinFBeta(A,I1A,I2A)
    Bbeta = oneMinFBeta(B,I1B,I2B)
    ABbeta = Abeta * Bbeta

    betas[0] = 1. - Abeta
    betas[1] = 1. - ABbeta
    betas[2] = 1. - Bbeta

@jit(nopython=True)
def simulate_lv_hcle(T : float, V : np.array, ks : np.array, save_at_ts : np.array, I1A : float, I2A : float, I1B : float, I2B : float, delta_t : float, Delta_t : float):

    props = np.zeros((3,))
    props_rint = np.zeros((3,))
    betas = np.zeros((3,))

    X = np.array([50.,60.])
    t = 0.

    X_hist = np.zeros((save_at_ts.size,2))

    i = 0
    while t < T:
        prop_fun(props, X, ks)
        prop_fun(props_rint, np.rint(X), ks)
        compute_betas(betas, X, I1A, I2A, I1B, I2B)
        #X,t = cs.hybrid1_tau_step_numba(X, t, V, props, props_rint, betas, delta_t, Delta_t)
        X,t = cs.hybrid1_cle_step_numba(X, t, V, props, props_rint, betas, delta_t, Delta_t)

        while save_at_ts[i] <= t and i < save_at_ts.size:
           X_hist[i] = X
           i+=1

    return X_hist, save_at_ts

def single_launch():
    rs,V,ks = define_system()
    T = 6.
    save_at_ts = np.linspace(0,T,1000)
    I1A,I2A,I1B,I2B = 5,10,5,10
    delta_t,Delta_t = 1e-2,1e-3
    X_hist, t_hist = simulate_lv_hcle(T, V, ks, save_at_ts, I1A, I2A, I1B, I2B, delta_t, Delta_t)

    plt.plot(t_hist, X_hist, '-o'); plt.show()


def MC_xp():
    N = 10000
    rs,V,ks = define_system()
    I1A,I2A,I1B,I2B = 5,10,5,10
    delta_t,Delta_t = 1e-2,1e-3
    T = 6.
    numsteps = 7
    #save_at_ts = np.linspace(0,T,numsteps)
    save_at_ts = np.arange(numsteps)
    dat = np.zeros((N,numsteps,2))
    for i in range(N):
        #if (i+1) % 1000 == 0:
        print(f'Iteration {i+1}')
        Xhist,thist = simulate_lv_hcle(T, V, ks, save_at_ts, I1A, I2A, I1B, I2B, delta_t, Delta_t)

        dat[i] = Xhist

    return dat,save_at_ts

def save_MC_dat(dat : np.array):
    np.save(f'lv/hcle-{dat.shape[0]}-{dat.shape[1]}', dat)

if __name__ == '__main__':
    #single_launch()
    dat,save_at_ts = MC_xp()
    #plt.plot(save_at_ts, dat.mean(axis=0)); plt.yscale('log'); plt.show()
    save_MC_dat(dat)
