import numpy as np
import chem_algs as ca
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
def simulate_lv_cle(T : float, V : np.array, ks : np.array, save_at_ts : np.array, delta_t : float):

    props = np.zeros((3,))
    X = np.array([50.,60.])
    t = 0.

    X_hist = np.zeros((save_at_ts.size,2))

    i = 0
    while t < T:
        prop_fun(props, X, ks)
        X,t = cs.cle_step_nb_refl(X, t, V, props, delta_t)

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
    X_hist, t_hist = simulate_lv_cle(T, V, ks, save_at_ts, delta_t)

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
        X_hist, t_hist = simulate_lv_cle(T, V, ks, save_at_ts, delta_t)

        dat[i] = X_hist

    return dat,save_at_ts

def save_MC_dat(dat : np.array):
    np.save(f'lv/cle-{dat.shape[0]}-{dat.shape[1]}', dat)

if __name__ == '__main__':
    #single_launch()
    dat,save_at_ts = MC_xp()
    plt.plot(save_at_ts, dat.mean(axis=0)); plt.yscale('log'); plt.show()
    save_MC_dat(dat)
