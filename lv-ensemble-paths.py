import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import sys
import matplotlib.pyplot as plt

'''
Produces files of format X{k}-{proc}.npy and corresponding t{i}-{proc}.npy using SSA for Lotka-Volterra system.
To be executed in parallel, and assembled to obtain histogram.
'''

t_abort = 30. # safety mechanism, discards few paths that run for too long 

N = int(sys.argv[1])
proc = int(sys.argv[2])

## set reactions
A, B = ['A'], ['B']
k1,k2,k3 = 2.,2e-3,2.

r1 = cu.Reaction(A, 2*A, k1)
r2 = cu.Reaction(A+B, 2*B, k2)
r3 = cu.Reaction(B, [], k3)
rs = cu.ReactionSet(r1, r2, r3)

V = rs.stochioimetric_matrix()
prop_func = lambda X : rs.propensity_f(X)

times = []

i=0
while i < N:
    t0 = time.time()

    X = np.array([50,60])

    T,t = 50.,0.
    X_hist, t_hist = [], []

    while t < T and time.time() - t0 < t_abort:
        X,t = cs.ssa_safe_step(X, t, V, prop_func)
        #X,t = cs.ssa_step(X, t, V, prop_func)
        X_hist.append(X)
        t_hist.append(t)

        print(t)
    t1 = time.time()

    times.append(t1-t0)

    X_hist = np.array(X_hist)
    t_hist = np.array(t_hist)

    np.save(f'X{i}-{proc}', X_hist)
    np.save(f't{i}-{proc}', t_hist)
    i+=1

# post-data geration, the resulting data of the paths is used to create a
# (scipy) sparse array (called 'hist-sparse.npz') M such that M[i,j] = sum_{realisations} #{X_t = [i,j]}
